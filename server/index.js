process.env.HF_ENDPOINT = 'https://hf-mirror.com';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'fs';
import { create, insert, search } from '@orama/orama';


// 尝试加载 .env.local，如果失败则使用硬编码
dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json({ limit: '4mb' }));

const state = {
  documents: [],
  chunks: []
};

// ============ 知识库持久化 ============
const STORAGE_DIR = path.resolve(__dirname, './kb_storage');
const DOCS_PATH = path.join(STORAGE_DIR, 'documents.json');
const CHUNKS_PATH = path.join(STORAGE_DIR, 'chunks.json');

function ensureStorageDir() {
  if (!fs.existsSync(STORAGE_DIR)) {
    fs.mkdirSync(STORAGE_DIR, { recursive: true });
  }
}

function saveKnowledgeBase() {
  try {
    ensureStorageDir();
    
    const docsMeta = state.documents.map(doc => ({
      id: doc.id,
      name: doc.name,
      createdAt: doc.createdAt
    }));
    fs.writeFileSync(DOCS_PATH, JSON.stringify(docsMeta, null, 2));
    fs.writeFileSync(CHUNKS_PATH, JSON.stringify(state.chunks, null, 2));
    
    console.log(`💾 知识库已保存: ${state.documents.length} 个文档, ${state.chunks.length} 个片段`);
  } catch (err) {
    console.error('❌ 保存知识库失败:', err.message);
  }
}

async function loadKnowledgeBase() {
  try {
    if (!fs.existsSync(CHUNKS_PATH)) {
      console.log('📂 未找到持久化数据，从空知识库开始');
      return false;
    }
    
    state.chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, 'utf-8'));
    
    if (fs.existsSync(DOCS_PATH)) {
      const docsMeta = JSON.parse(fs.readFileSync(DOCS_PATH, 'utf-8'));
      state.documents = docsMeta.map(meta => ({
        ...meta,
        content: ''
      }));
    } else {
      const docMap = new Map();
      for (const chunk of state.chunks) {
        if (!docMap.has(chunk.documentId)) {
          docMap.set(chunk.documentId, {
            id: chunk.documentId,
            name: chunk.documentName,
            createdAt: Date.now(),
            content: ''
          });
        }
      }
      state.documents = Array.from(docMap.values());
    }
    
    await buildBM25Index();
    console.log(`📂 知识库已加载: ${state.documents.length} 个文档, ${state.chunks.length} 个片段`);
    return true;
  } catch (err) {
    console.error('❌ 加载知识库失败:', err.message);
    return false;
  }
}

// ============ 硬编码 API Key（临时方案）============
const HARDCODED_API_KEY = 'sk-18129c710f2d44d7b9101f2b7093778e';

const config = {
  apiKey: process.env.QWEN_API_KEY || HARDCODED_API_KEY,
  baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  model: process.env.QWEN_MODEL || 'deepseek-v4-flash',
  embeddingModel: 'text-embedding-v1',
  rerankerModel: 'gte-rerank-v2',
  port: Number(process.env.SERVER_PORT || 8787)
};

console.log(`🔑 API Key 状态: ${config.apiKey ? '已配置 (前8位: ' + config.apiKey.slice(0, 8) + '...)' : '❌ 未配置'}`);

// ============ 工具函数 ============

function uid(prefix) {
  return `${prefix}-${crypto.randomUUID()}`;
}

function createAppError(code, message, details = '', status = 500) {
  const error = new Error(message);
  error.code = code;
  error.details = details;
  error.status = status;
  return error;
}

function getErrorPayload(error, fallbackMessage) {
  if (error && typeof error === 'object' && 'message' in error) {
    return {
      code: error.code || 'UNKNOWN_ERROR',
      message: error.message || fallbackMessage,
      details: error.details || '',
      status: error.status || 500
    };
  }
  return { code: 'UNKNOWN_ERROR', message: fallbackMessage, details: '', status: 500 };
}

function tokenize(input) {
  return input
    .toLowerCase()
    .replace(/[`*_>#\-\[\]\(\)]/g, ' ')
    .split(/[\s，。；：！？、,.!?;:\/\\|]+/)
    .filter((token) => token.length > 1);
}

// ============ 优化1：递归分块 ============

function chunkText(text, chunkSize = 1200, overlap = 200) {
  const separators = ["\n\n", "\n", "。", "；", "？", "！", ".", ";", " "];

  function splitRecursive(text, sepIndex = 0) {
    if (text.length <= chunkSize) return [text.trim()];
    if (sepIndex >= separators.length) {
      const chunks = [];
      let start = 0;
      while (start < text.length) {
        const end = Math.min(text.length, start + chunkSize);
        chunks.push(text.slice(start, end).trim());
        start += chunkSize - overlap;
      }
      return chunks;
    }
    const separator = separators[sepIndex];
    const parts = text.split(separator);
    const chunks = [];
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i].trim();
      if (part.length === 0) continue;
      if (i < parts.length - 1) {
        const combined = part + separator + parts[i + 1].trim();
        if (combined.length <= chunkSize) {
          parts[i + 1] = combined;
          continue;
        }
      }
      if (part.length > chunkSize) {
        chunks.push(...splitRecursive(part, sepIndex + 1));
      } else {
        chunks.push(part);
      }
    }
    return chunks;
  }
  return splitRecursive(text).filter(c => c.length > 0);
}

function cosineSimilarity(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (!normA || !normB) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ============ 优化2：混合检索（Orama BM25 + 向量）============

let oramaDB = null;

async function buildBM25Index() {
  if (state.chunks.length === 0) {
    oramaDB = null;
    return;
  }
  oramaDB = await create({
    schema: { text: 'string', id: 'string' }
  });
  for (const chunk of state.chunks) {
    await insert(oramaDB, { text: chunk.text, id: chunk.id });
  }
  console.log(`✅ Orama BM25 索引构建完成，共 ${state.chunks.length} 个文档`);
}

async function searchByKeyword(query, topK = 30) {
  if (!oramaDB || state.chunks.length === 0) return [];
  const results = await search(oramaDB, { term: query, limit: topK });
  return results.hits.map(hit => ({
    chunk: state.chunks.find(c => c.id === hit.document.id),
    score: hit.score,
  }));
}
// ============ 知识图谱服务调用 ============
const KG_URL = 'http://127.0.0.1:5000';

/**
 * 上传文档后，调用 Python 服务构建知识图谱。
 * 在 ingestDocuments 完成后调用。
 */
async function buildKnowledgeGraph() {
  try {
    const chunks = state.chunks.map(c => ({
      id: c.id,
      text: c.rawText || c.text,
      doc_name: c.documentName
    }));
    
    const response = await fetch(`${KG_URL}/build_graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunks })
    });
    const result = await response.json();
    console.log(`🧠 知识图谱构建完成: ${result.node_count} 实体, ${result.edge_count} 关系`);
  } catch (error) {
    console.error('⚠️ 知识图谱构建失败:', error.message);
  }
}

/**
 * 检索时调用图谱查询，获取证据链
 */
async function graphQuery(query) {
  try {
    const response = await fetch(`${KG_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, max_hops: 1 })
    });
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('⚠️ 图谱查询失败:', error.message);
    return { evidence: [], context: '' };
  }
}
function rrfFusion(vectorResults, keywordResults, topK, k = 60) {
  const scores = new Map();
  vectorResults.forEach((item, rank) => {
    const id = item.chunk.id;
    scores.set(id, { chunk: item.chunk, score: (1 / (k + rank + 1)) * 0.7 });
  });
  keywordResults.forEach((item, rank) => {
    const id = item.chunk.id;
    const rrfScore = 1 / (k + rank + 1);
    const existing = scores.get(id);
    if (existing) {
      existing.score += rrfScore * 0.3;
    } else {
      scores.set(id, { chunk: item.chunk, score: rrfScore * 0.3 });
    }
  });
  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

// ============ 优化3：重排序 ============

async function rerankFetch(query, documents, topK = 5) {
    console.log(`🔄 重排序: ${documents.length} 个文档 → ${topK} 个`);
    try {
        const response = await fetch(
            'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank',
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${config.apiKey}`
                },
                body: JSON.stringify({
                    model: 'gte-rerank',
                    input: {
                        query: query,
                        documents: documents.map(doc =>
                            String(doc.chunk?.text ?? '').slice(0, 4000)
                        )
                    },
                    parameters: { top_n: topK }
                })
            }
        );

        if (!response.ok) {
            const text = await response.text().catch(() => '');
            console.error(`重排序错误 ${response.status}:`, text);
            return null;
        }

        const result = await response.json();
        
        if (result?.output?.results) {
            return result.output.results;
        }

        console.error('重排序响应格式异常:', result);
        return null;

    } catch (error) {
        console.error(`重排序失败: ${error.message}`);
        return null;
    }
}

async function rerankDocuments(query, documents, topK = 5) {
    if (documents.length <= topK) return documents;

    const results = await rerankFetch(query, documents, topK);

    if (results && Array.isArray(results)) {
        return results
            .map(item => ({
                ...documents[item.index],
                rerankScore: item.relevance_score
            }))
            .sort((a, b) => b.rerankScore - a.rerankScore);
    }

    return documents.slice(0, topK);
}

// ============ API 调用封装 ============

async function qwenFetch(endpoint, body, stream = false) {
  if (!config.apiKey) {
    throw createAppError('MISSING_API_KEY', '缺少 Qwen API Key', '请检查服务端配置。', 500);
  }
  let response;
  try {
    response = await fetch(`${config.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${config.apiKey}` },
      body: JSON.stringify(body)
    });
  } catch {
    throw createAppError('NETWORK_UNREACHABLE', '无法连接到 Qwen 服务', '网络连接失败', 502);
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    if (response.status === 401) throw createAppError('INVALID_API_KEY', 'Qwen API Key 无效或已过期', text, 401);
    if (response.status === 429) throw createAppError('RATE_LIMITED', 'Qwen 请求过于频繁', text, 429);
    throw createAppError('QWEN_HTTP_ERROR', `Qwen 请求失败（${response.status}）`, text, 502);
  }
  if (stream) return response;
  return response.json();
}

async function createEmbedding(text) {
  const result = await qwenFetch('/embeddings', {
    model: config.embeddingModel,
    input: text.slice(0, 6000)
  });
  const vector = result.data?.[0]?.embedding;
  if (!vector) throw createAppError('EMBEDDING_EMPTY', 'Embedding 生成失败', '模型返回为空', 502);
  return vector;
}

// ============ 查询改写 ============

async function rewriteQuery(originalQuery, conversationHistory = []) {
  console.log(`🔄 rewriteQuery 被调用: "${originalQuery.slice(0, 50)}..."`);
  try {
    const prompt = `你是一个查询改写助手。将用户的口语化、模糊或依赖上下文的问题改写为一个独立、完整、精准的检索查询。

对话历史：
${conversationHistory.slice(-3).map(m => `${m.role}: ${m.content}`).join('\n')}

用户当前问题：${originalQuery}

请输出改写后的查询（只输出查询内容，不要解释）：`;
    const completion = await qwenFetch('/chat/completions', {
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 200
    });
    const rewritten = completion.choices[0].message.content.trim();
    console.log(`✅ 改写成功: "${rewritten}"`);
    return rewritten;
  } catch (error) {
    console.error(`❌ 查询改写失败: ${error.message}，使用原问题`);
    return originalQuery;
  }
}

// ============ 实体标签生成 ============

async function extractEntities(text) {
  const prompt = `你是一个学术论文分析助手。请从以下论文片段中，提取**能够帮助跨论文关联的关键概念标签**。

提取原则：
1. 模型/方法名称（如 OKH-RAG、ACTD、DPR）
2. 核心研究主题（如 检索增强生成、知识图谱、混合检索）
3. 解决的核心问题（如 幻觉抑制、选择性检索、多跳推理）
4. 重要的技术范式（如 轨迹推理、顺序感知超图）

不要提取：
- 过于细小的实验细节
- 具体的数值或评估指标
- 通用技术术语（除非是论文核心主题）

请只输出一个 JSON 字符串数组，包含 3-5 个最重要的标签。
例如：["OKH-RAG", "轨迹推理", "顺序感知超图"]

文本：
${text.slice(0, 2000)}`;

  try {
    const completion = await qwenFetch('/chat/completions', {
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens: 300
    });
    const raw = completion.choices[0].message.content.trim();
    const entities = JSON.parse(raw);
    console.log(`🏷️  生成标签: ${entities.length} 个 → ${entities.join(', ')}`);
    return Array.isArray(entities) ? entities : [];
  } catch (error) {
    console.error(`⚠️ 标签生成失败: ${error.message}`);
    return [];
  }
}

// ============ 检索核心函数 ============

function buildCitation(chunk, score) {
  return {
    id: chunk.id,
    title: chunk.documentName,
    snippet: chunk.rawText ? chunk.rawText.slice(0, 300) : chunk.text.slice(0, 300),
    source: `向量知识库 / ${chunk.documentName}`,
    score: Number(score.toFixed(4)),
    entities: chunk.entities || []
  };
}

async function searchKnowledge(query, topK = 5, conversationHistory = []) {
  if (!state.chunks.length) return [];

  console.log(`🔍 searchKnowledge 被调用: "${query.slice(0, 50)}..."`);

  // ① 查询改写
  let rewrittenQuery = query;
  try {
    rewrittenQuery = await rewriteQuery(query, conversationHistory);
    if (rewrittenQuery !== query) {
      console.log(`✅ 查询改写: "${query.slice(0, 30)}..." → "${rewrittenQuery.slice(0, 30)}..."`);
    }
  } catch (error) {
    console.error(`❌ 查询改写失败: ${error.message}，使用原问题`);
  }

  // ② 提取查询标签
  const queryEntities = await extractEntities(rewrittenQuery);
  console.log(`🔍 查询标签: ${queryEntities.join(', ')}`);

  // ③ 标签粗筛
  let candidates = state.chunks;
  if (queryEntities.length > 0) {
    const taggedCandidates = state.chunks.filter(chunk => {
      const chunkTags = chunk.entities || [];
      return queryEntities.some(qTag =>
        chunkTags.some(cTag => cTag.toLowerCase().includes(qTag.toLowerCase()))
      );
    });
    if (taggedCandidates.length >= topK) {
      candidates = taggedCandidates;
      console.log(`📋 标签召回: ${candidates.length} 个候选文档`);
    } else {
      console.log(`⚠️ 标签召回不足 (${taggedCandidates.length}/${topK})，降级为全量检索`);
    }
  }

  // ④ 向量检索
  const queryEmbedding = await createEmbedding(rewrittenQuery);
  const vectorResults = candidates
    .map(chunk => ({ chunk, score: cosineSimilarity(queryEmbedding, chunk.embedding) }))
    .filter(item => item.score > 0.15)
    .sort((a, b) => b.score - a.score)
    .slice(0, 20);

  // ⑤ BM25 关键词召回
  const keywordResults = await searchByKeyword(rewrittenQuery, 20);

  // ⑥ RRF 融合
  const fusedResults = rrfFusion(vectorResults, keywordResults, 20);

  // ⑦ 重排序
   // ⑦ 重排序
  const rerankedResults = await rerankDocuments(rewrittenQuery, fusedResults, topK);

  // ⑧ 知识图谱查询（新增）
  try {
    const graphResult = await graphQuery(rewrittenQuery);
    if (graphResult.evidence && graphResult.evidence.length > 0) {
      console.log(`🧠 图谱证据: ${graphResult.evidence.length} 条`);
      
      // 将图谱证据链封装成 citation 格式，与向量检索结果合并
      const graphChunks = graphResult.evidence.map(e => ({
        chunk: {
          id: `graph_${e.subject}_${e.object}`,
          documentName: '知识图谱',
          rawText: `${e.subject} —[${e.predicate}]→ ${e.object}`,
          text: `${e.subject} —[${e.predicate}]→ ${e.object}`
        },
        score: 0.9
      }));
      
      // 合并去重
      const allResults = [...rerankedResults, ...graphChunks];
      const seen = new Set();
      const merged = [];
      for (const item of allResults) {
        const key = (item.chunk.rawText || item.chunk.text).slice(0, 30);
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(item);
        }
      }
      
      return merged.slice(0, topK).map(({ chunk, score }) => buildCitation(chunk, score));
    }
  } catch (error) {
    console.error('⚠️ 图谱查询失败:', error.message);
  }

  return rerankedResults.map(({ chunk, score }) => buildCitation(chunk, score));
}

// ============ 文档上传与索引 ============

let pdfParse = null;
async function getPdfParse() {
  if (pdfParse) return pdfParse;
  try {
    const module = await import('pdf-parse');
    pdfParse = module.default;
    console.log('✅ PDF 解析库加载成功');
  } catch (err) {
    console.error('⚠️ PDF 解析库加载失败:', err.message);
  }
  return pdfParse;
}

async function ingestDocuments(files) {
  const supported = files.filter((file) =>
    /\.(txt|md|markdown|json|pdf)$/i.test(file.originalname)
  );
  if (!supported.length) {
    throw createAppError('UNSUPPORTED_FILES', '没有可导入的知识文件',
      '仅支持 .md、.markdown、.txt、.json、.pdf 文件', 400);
  }

  const inserted = [];
  const nextDocuments = [];
  const nextChunks = [];

  for (const file of supported) {
    let content = '';
    if (/\.pdf$/i.test(file.originalname)) {
      const parser = await getPdfParse();
      if (parser) {
        try {
          const pdfData = await parser(file.buffer);
          content = pdfData.text.trim();
        } catch (err) {
          continue;
        }
      } else {
        continue;
      }
    } else {
      content = file.buffer.toString('utf-8').trim();
    }
    if (!content) continue;

    const document = {
      id: uid('doc'),
      name: file.originalname,
      content: content,
      createdAt: Date.now()
    };
    nextDocuments.push(document);
    inserted.push(document);

    const parts = chunkText(content);
    console.log(`📝 ${file.originalname} 切分为 ${parts.length} 个片段`);

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];

      console.log(`  🏷️  生成实体标签 ${i + 1}/${parts.length}...`);
      const entities = await extractEntities(part);
      await new Promise(resolve => setTimeout(resolve, 500));

      const tagPrefix = entities.length > 0
        ? `[标签: ${entities.join(', ')}]\n`
        : '';
      const enrichedText = tagPrefix + part;

      console.log(`  🔄 生成 Embedding ${i + 1}/${parts.length}...`);
      const embedding = await createEmbedding(enrichedText);

      nextChunks.push({
        id: uid('chunk'),
        documentId: document.id,
        documentName: document.name,
        text: enrichedText,
        rawText: part,
        tokens: tokenize(part),
        embedding,
        entities
      });
    }
  }

  if (!inserted.length) {
    throw createAppError('EMPTY_FILES', '上传的文件内容为空', '请确认文件不是空文件', 400);
  }

  state.documents.push(...nextDocuments);
  state.chunks.push(...nextChunks);
  await buildBM25Index();
  saveKnowledgeBase();
  await buildKnowledgeGraph();
  console.log(`✅ 知识库更新: ${inserted.length} 个文档, ${nextChunks.length} 个片段`);
  return inserted;
}

// ============ SSE 与工具调用 ============

function sendSse(res, event, data) {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function createToolDefinitions() {
  return [
    {
      type: 'function',
      function: {
        name: 'retrieve_knowledge',
        description: '从后端向量知识库检索与用户问题最相关的文档片段',
        parameters: {
          type: 'object',
          properties: {
            query: { type: 'string', description: '需要检索的查询内容' },
            topK: { type: 'integer', description: '返回结果数量，默认 4' }
          },
          required: ['query']
        }
      }
    },
    {
      type: 'function',
      function: {
        name: 'list_knowledge_documents',
        description: '列出当前后端知识库中的文档',
        parameters: { type: 'object', properties: {} }
      }
    },
    {
      type: 'function',
      function: {
        name: 'get_current_time',
        description: '获取当前系统时间',
        parameters: { type: 'object', properties: {} }
      }
    }
  ];
}

async function executeToolCall(toolCall, messages = []) {
  const name = toolCall.function?.name || 'unknown';
  const args = toolCall.function?.arguments ? JSON.parse(toolCall.function.arguments) : {};

  if (name === 'retrieve_knowledge') {
    const citations = await searchKnowledge(
      String(args.query || ''),
      Number(args.topK || 4),
      messages
    );
    let contextText = citations.length === 0
      ? '未找到相关信息'
      : citations.map((c, i) =>
          `[${i + 1}] 来自文件 "${c.title}":\n${c.snippet}`
        ).join('\n\n');
    return {
      result: { success: true, count: citations.length, context: contextText },
      citations
    };
  }

  if (name === 'list_knowledge_documents') {
    return {
      result: {
        documents: state.documents.map(doc => ({ id: doc.id, name: doc.name, createdAt: doc.createdAt }))
      },
      citations: []
    };
  }

  if (name === 'get_current_time') {
    return {
      result: {
        iso: new Date().toISOString(),
        locale: new Date().toLocaleString('zh-CN', { hour12: false })
      },
      citations: []
    };
  }

  return { result: { error: '未知工具' }, citations: [] };
}

// ============ API 路由 ============

app.get('/api/health', (_, res) => {
  res.json({ ok: true, documents: state.documents.length, chunks: state.chunks.length });
});

app.get('/api/knowledge', (_, res) => {
  res.json({
    documents: state.documents.map(doc => ({ id: doc.id, name: doc.name, createdAt: doc.createdAt }))
  });
});

app.post('/api/knowledge/upload', upload.array('files'), async (req, res) => {
  try {
    const files = Array.isArray(req.files) ? req.files : [];
    const inserted = await ingestDocuments(files);
    res.json({
      documents: inserted.map(doc => ({ id: doc.id, name: doc.name, createdAt: doc.createdAt })),
      message: `已成功导入 ${inserted.length} 份知识文件。`
    });
  } catch (error) {
    const payload = getErrorPayload(error, '知识库导入失败');
    res.status(payload.status).json({ error: payload.message, code: payload.code, details: payload.details });
  }
});

app.delete('/api/knowledge/:id', (req, res) => {
  const { id } = req.params;
  state.documents = state.documents.filter(doc => doc.id !== id);
  state.chunks = state.chunks.filter(chunk => chunk.documentId !== id);
  buildBM25Index();
  saveKnowledgeBase();
  res.json({ ok: true });
});

app.delete('/api/knowledge', (_, res) => {
  state.documents = [];
  state.chunks = [];
  oramaDB = null;
  saveKnowledgeBase();
  res.json({ ok: true });
});

app.post('/api/chat/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');

  try {
    const incomingMessages = Array.isArray(req.body?.messages) ? req.body.messages : [];
    const tools = createToolDefinitions();
    const messages = [
      {
        role: 'system',
        content: [
          '你是一个专业的知识问答助手。',
          '',
          '【知识库信息】',
          state.documents.length > 0
            ? `当前知识库包含以下文档：${state.documents.map(d => d.name).join('、')}`
            : '当前知识库为空。',
          '',
          '【回答策略——请严格遵循】',
          '1. 优先调用 retrieve_knowledge 检索相关文档',
          '2. 基于检索结果回答，控制在300字以内，只讲核心要点',
          '3. 即使检索结果不完整，也要基于已有信息给出部分答案，不要直接说"无法回答"',
          '4. 只有在检索结果完全无关联时，才说明信息不足',
          '5. 回答结构：先用两三句话给出核心结论，再用一两句话补充细节依据',
          '6. 不要逐句分析检索内容，不要重复列举已检索到的上下文'
        ].join('\n')
      },
      ...incomingMessages
    ];

    const gatheredCitations = [];
    const gatheredTools = [];

    for (let round = 0; round < 4; round++) {
      const completion = await qwenFetch('/chat/completions', {
        model: config.model,
        stream: false,
        temperature: 0.1,
        messages,
        tools,
        tool_choice: 'auto'
      });

      const choice = completion.choices?.[0]?.message;
      const toolCalls = choice?.tool_calls || [];

      if (!toolCalls.length) {
        if (choice?.content) sendSse(res, 'token', { token: choice.content });
        break;
      }

      messages.push({ role: 'assistant', content: choice.content || '', tool_calls: toolCalls });

      for (const toolCall of toolCalls) {
        const args = toolCall.function?.arguments ? JSON.parse(toolCall.function.arguments) : {};
        sendSse(res, 'tool', { id: toolCall.id, name: toolCall.function?.name, args, status: 'running' });

        const executed = await executeToolCall(toolCall, messages);
        gatheredCitations.push(...executed.citations);
        gatheredTools.push({
          id: toolCall.id, name: toolCall.function?.name, args,
          status: 'success', result: JSON.stringify(executed.result, null, 2)
        });

        sendSse(res, 'tool', { id: toolCall.id, name: toolCall.function?.name, args, status: 'success', result: executed.result });
        if (executed.citations.length) sendSse(res, 'citations', { citations: executed.citations });

        let toolContent = executed.result.context || JSON.stringify(executed.result.citations || executed.result);
        messages.push({ role: 'tool', tool_call_id: toolCall.id, content: toolContent });
      }
    }

    const streamResponse = await qwenFetch('/chat/completions', {
      model: config.model, stream: true, temperature: 0.1, messages
    }, true);

    const reader = streamResponse.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';

      for (const event of events) {
        const lines = event.split('\n').map(l => l.trim()).filter(Boolean);
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const raw = line.slice(5).trim();
          if (raw === '[DONE]') {
            sendSse(res, 'done', { citations: gatheredCitations, tools: gatheredTools });
            res.end();
            return;
          }
          try {
            const json = JSON.parse(raw);
            const token = json.choices?.[0]?.delta?.content;
            if (token) sendSse(res, 'token', { token });
          } catch {}
        }
      }
    }

    sendSse(res, 'done', { citations: gatheredCitations, tools: gatheredTools });
    res.end();
  } catch (error) {
    const payload = getErrorPayload(error, '服务异常');
    sendSse(res, 'error', { code: payload.code, message: payload.message, details: payload.details });
    res.end();
  }
});
// ============ 知识图谱构建接口 ============
app.post('/api/kg/build', async (req, res) => {
  try {
    console.log('🧠 手动触发知识图谱构建...');
    await buildKnowledgeGraph(); // 直接调用你已有的构建函数
    res.json({ message: '知识图谱构建完成' });
  } catch (error) {
    console.error('知识图谱构建失败:', error.message);
    res.status(500).json({ error: error.message });
  }
});
// ============ 评估数据收集 ============

app.post('/api/collect-eval-data', async (req, res) => {
  const { queries } = req.body;
  if (!Array.isArray(queries) || queries.length === 0) {
    return res.status(400).json({ error: 'queries 必须是非空数组' });
  }

  const results = [];
  for (const query of queries) {
    console.log(`🔄 处理问题: ${query.slice(0, 50)}...`);
    try {
  // ✅ 新增：先查知识图谱
  let graphContext = '';
  try {
    const graphResult = await graphQuery(query);
    if (graphResult.context && graphResult.context.length > 0) {
      graphContext = '\n\n【知识图谱证据链】\n' + graphResult.context;
      console.log(`🧠 图谱注入: ${graphResult.evidence?.length || 0} 条证据`);
    }
  } catch (e) {
    console.error('图谱查询失败:', e.message);
  }

  // 原有的向量检索
  const citations = await searchKnowledge(query, 4);
  const context = citations.length > 0 
    ? citations.map(c => c.snippet).join('\n\n')
    : '未找到相关信息';
  
  // ✅ 合并上下文：图谱证据 + 向量检索结果
  const fullContext = graphContext + '\n\n' + context;
  
  const completion = await qwenFetch('/chat/completions', {
    model: config.model,
    messages: [
      { role: 'system', content: '你是一个专业的知识问答助手。请严格基于提供的上下文回答问题，如果上下文没有相关信息，请诚实说明。' },
      { role: 'user', content: `上下文：\n${fullContext}\n\n问题：${query}` }
    ],
    temperature: 0.1
  });

  const answer = completion.choices?.[0]?.message?.content || '';
  results.push({ question: query, answer, contexts: citations.map(c => c.snippet), ground_truth: "" });
  console.log(`   ✅ 完成，答案长度: ${answer.length} 字符`);
} catch (error) {
  console.error(`   ❌ 处理失败: ${error.message}`);
      results.push({ question: query, answer: `处理失败: ${error.message}`, contexts: [], ground_truth: "" });
}
   
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  fs.writeFileSync('./eval_data.json', JSON.stringify(results, null, 2));
  console.log(`\n✅ 已保存 ${results.length} 条评估数据到 eval_data.json`);
  res.json({ message: `已保存 ${results.length} 条评估数据`, results });
});

// ============ 静态文件托管 ============

const distPath = path.resolve(__dirname, '../dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get('*', (_, res) => res.sendFile(path.join(distPath, 'index.html')));
} else {
  console.log('⚠️ dist 目录不存在，跳过静态文件托管（API仍正常工作）');
  app.get('*', (_, res) => res.json({ message: 'API服务正常运行，前端请访问开发服务器' }));
}

// ============ 启动服务 ============

async function startServer() {
  await loadKnowledgeBase();
  
  app.listen(config.port, '127.0.0.1', () => {
    console.log(`Server running at http://127.0.0.1:${config.port}`);
  });
}

startServer();