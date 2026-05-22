from flask import Flask, request, jsonify, send_from_directory
import networkx as nx
import time  # ← 加在文件最顶部，和其他 import 放一起
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from collections import deque
MODEL_LIST = ['qwen3.6-plus', 'qwen-plus-latest', 'qwen-plus-0112', 'qwen3-8b','qwen2.5-14b-instruct-1m','deepseek-r1']
_current_model_idx = 0
app = Flask(__name__)
def get_current_model():
    return MODEL_LIST[_current_model_idx]

def switch_to_next_model():
    global _current_model_idx
    _current_model_idx = (_current_model_idx + 1) % len(MODEL_LIST)
    new_model = MODEL_LIST[_current_model_idx]
    print(f'⚠️ 模型自动切换: {MODEL_LIST[_current_model_idx - 1]} → {new_model}')
    return new_model

# ==================== 全局变量 ====================
kg_graph = nx.DiGraph()
entity_index = {}
triplet_store = []
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
GRAPH_STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'graph_storage')
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(GRAPH_STORAGE_DIR, exist_ok=True)


# ==================== 图谱持久化 ====================

def save_graph():
    """将知识图谱持久化到磁盘"""
    global kg_graph, triplet_store
    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        return
    
    with open(os.path.join(GRAPH_STORAGE_DIR, 'triplets.json'), 'w', encoding='utf-8') as f:
        json.dump(triplet_store, f, ensure_ascii=False, indent=2)
    
    graph_data = {
        'nodes': list(kg_graph.nodes()),
        'edges': [(u, v, d) for u, v, d in kg_graph.edges(data=True)]
    }
    with open(os.path.join(GRAPH_STORAGE_DIR, 'graph.json'), 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f'💾 知识图谱已保存: {kg_graph.number_of_nodes()} 实体, {kg_graph.number_of_edges()} 关系')


def load_graph():
    """从磁盘加载知识图谱"""
    global kg_graph, entity_index, triplet_store
    
    triplets_path = os.path.join(GRAPH_STORAGE_DIR, 'triplets.json')
    graph_path = os.path.join(GRAPH_STORAGE_DIR, 'graph.json')
    
    if not os.path.exists(triplets_path) or not os.path.exists(graph_path):
        print('📂 未找到图谱持久化数据，从空图谱开始')
        return False
    
    try:
        with open(triplets_path, 'r', encoding='utf-8') as f:
            triplet_store = json.load(f)
        
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        kg_graph = nx.DiGraph()
        kg_graph.add_nodes_from(graph_data['nodes'])
        for u, v, d in graph_data['edges']:
            kg_graph.add_edge(u, v, **d)
        
        entity_index = {node: i for i, node in enumerate(kg_graph.nodes())}
        
        print(f'📂 知识图谱已加载: {kg_graph.number_of_nodes()} 实体, {kg_graph.number_of_edges()} 关系')
        return True
    except Exception as e:
        print(f'❌ 加载知识图谱失败: {e}')
        return False


# ==================== LLM 调用封装 ====================

def call_llm(prompt, api_key=None, base_url=None, model=None):
    import requests
    import time
    
    api_key = api_key or os.environ.get('QWEN_API_KEY', 'sk-18129c710f2d44d7b9101f2b7093778e')
    base_url = base_url or os.environ.get('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    
    if not model:
        model = get_current_model()  # ✅ 使用模型列表中的当前模型
    
    if not api_key:
        raise ValueError('请在环境变量中设置 QWEN_API_KEY')
    
    # ✅ 最多重试整个模型列表的长度次数
    max_retries = len(MODEL_LIST)
    
    for attempt in range(max_retries):
        try:
            print(f'  🤖 当前模型: {model}')
            response = requests.post(
                f'{base_url}/chat/completions',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                },
                json={
                    'model': model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1,
                    'max_tokens': 2000
                },
                timeout=180
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            
            # ✅ 额度耗尽 → 切换下一个模型
            elif response.status_code == 403 or response.status_code == 429:
                print(f'  ⚠️ 模型 {model} 额度耗尽或限流，切换下一个...')
                model = switch_to_next_model()
                time.sleep(2)
                
            else:
                print(f'  ❌ 请求失败 ({response.status_code}): {response.text[:200]}')
                model = switch_to_next_model()
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            print(f'  ⏳ 模型 {model} 超时，切换下一个...')
            model = switch_to_next_model()
            time.sleep(2)
            
        except requests.exceptions.ConnectionError:
            print(f'  🔌 连接错误，等待 5 秒后重试...')
            time.sleep(5)
    
    raise Exception('LLM 请求失败：所有模型均已尝试')
# ==================== 三元组抽取 ====================

def extract_triplets(text):
    prompt = f"""你是一个学术论文知识抽取助手。请从以下论文片段中，抽取核心的**实体关系三元组**。

规则：
1. 每个三元组格式为：`[实体1] | [关系] | [实体2]`
2. 关系必须是**明确的学术语义关系**，例如：
   - `提出` / `改进` / `推翻` / `属于` / `用于` / `对比` / `优于` / `结合` / `解决`
3. 实体应该是**有意义的概念、方法、模型、问题**
4. 尽量抽取 3-8 个最重要的三元组

输出格式（每行一个三元组，不要编号，不要其他内容）：
OKH-RAG | 提出 | 轨迹推理
轨迹推理 | 解决 | 多跳推理问题

论文片段：
{text[:3000]}
"""
    
    try:
        response = call_llm(prompt)
        triplets = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3 and all(parts):
                    triplets.append({
                        'subject': parts[0],
                        'predicate': parts[1],
                        'object': parts[2]
                    })
        return triplets
    except Exception as e:
        print(f'三元组抽取失败: {e}')
        return []


# ==================== 图谱构建 ====================

@app.route('/build_graph', methods=['POST'])
def build_graph():
    global kg_graph, entity_index, triplet_store
    
    data = request.json
    chunks = data.get('chunks', [])
    
    if not chunks:
        return jsonify({'error': 'chunks 不能为空'}), 400
    
    kg_graph = nx.DiGraph()
    entity_index = {}
    triplet_store = []
    
    print(f'🔄 开始处理 {len(chunks)} 个文本块...')
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        doc_name = chunk.get('doc_name', 'unknown')
        
        print(f'  📄 处理块 {i+1}/{len(chunks)}: {doc_name[:50]}...')
        time.sleep(3)  # ← 加这行
        
        triplets = extract_triplets(text)
        
        for triplet in triplets:
            subject = triplet['subject']
            predicate = triplet['predicate']
            obj = triplet['object']
            
            for entity in [subject, obj]:
                if entity not in entity_index:
                    entity_index[entity] = len(entity_index)
                    kg_graph.add_node(entity, name=entity)
            
            kg_graph.add_edge(subject, obj, relation=predicate)
            
            triplet_store.append({
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'source_chunk': chunk.get('id', f'chunk_{i}')
            })
        
        print(f'      抽取到 {len(triplets)} 个三元组')
    
    save_graph()
    
    print(f'✅ 知识图谱构建完成: {kg_graph.number_of_nodes()} 个实体, '
          f'{kg_graph.number_of_edges()} 条关系, {len(triplet_store)} 个三元组')
    
    return jsonify({
        'status': 'ok',
        'node_count': kg_graph.number_of_nodes(),
        'edge_count': kg_graph.number_of_edges(),
        'triplet_count': len(triplet_store)
    })


# ==================== 图谱查询 ====================

@app.route('/query', methods=['POST'])
def query_graph():
    global kg_graph, triplet_store
    
    data = request.json
    query = data.get('query', '')
    query_entities = data.get('entities', None)
    max_hops = data.get('max_hops', 1)
    
    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        return jsonify({'evidence': [], 'context': '', 'matched_entities': []})
    
    # 提取查询实体
    if query_entities is None:
        query_entities = extract_query_entities(query)
    
    print(f'🔍 查询: "{query}"')
    print(f'📌 识别实体: {query_entities}')
    
    # 在图中匹配实体
    matched_entities = []
    for q_entity in query_entities:
        q_lower = q_entity.lower()
        for entity in kg_graph.nodes():
            if q_lower in entity.lower() or entity.lower() in q_lower:
                if entity not in matched_entities:
                    matched_entities.append(entity)
                    break
    
    # 如果精确匹配失败，尝试部分匹配
    if not matched_entities:
        for q_entity in query_entities:
            for entity in kg_graph.nodes():
                if any(word.lower() in entity.lower() for word in q_entity.split()):
                    if entity not in matched_entities:
                        matched_entities.append(entity)
    
    print(f'🎯 匹配到实体: {matched_entities}')
    
    # 子图遍历
    evidence = []
    visited = set()
    queue = deque()
    
    for entity in matched_entities:
        queue.append((entity, 0))
        visited.add(entity)
    
    while queue:
        current, hops = queue.popleft()
        if hops >= max_hops:
            continue
        
        for neighbor in kg_graph.successors(current):
            edge_data = kg_graph.get_edge_data(current, neighbor)
            relation = edge_data.get('relation', '相关')
            evidence.append({
                'subject': current,
                'predicate': relation,
                'object': neighbor
            })
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, hops + 1))
        
        for predecessor in kg_graph.predecessors(current):
            edge_data = kg_graph.get_edge_data(predecessor, current)
            relation = edge_data.get('relation', '相关')
            evidence.append({
                'subject': predecessor,
                'predicate': relation,
                'object': current
            })
            if predecessor not in visited:
                visited.add(predecessor)
                queue.append((predecessor, hops + 1))
    
    # 去重
    seen = set()
    unique_evidence = []
    for e in evidence:
        key = (e['subject'], e['predicate'], e['object'])
        if key not in seen:
            seen.add(key)
            unique_evidence.append(e)
    
    # 构建上下文字符串
    context_lines = []
    for e in unique_evidence:
        is_relevant = any(
            ent.lower() in e['subject'].lower() or ent.lower() in e['object'].lower()
            for ent in query_entities
        )
        if is_relevant:
            context_lines.append(f"{e['subject']} —[{e['predicate']}]→ {e['object']}")
    
    context = '知识图谱证据链：\n' + '\n'.join(context_lines) if context_lines else ''
    
    print(f'📊 找到 {len(unique_evidence)} 条证据')
    
    return jsonify({
        'evidence': unique_evidence,
        'context': context,
        'matched_entities': matched_entities,
        'related_entities': list(visited - set(matched_entities)),
        'total_nodes': kg_graph.number_of_nodes(),
        'total_edges': kg_graph.number_of_edges()
    })


# ==================== 查询实体提取 ====================

def extract_query_entities(query):
    prompt = f"""从以下用户问题中，提取出可以作为知识图谱检索起点的**关键实体**（如方法名、模型名、技术概念）。
只输出实体名称，每行一个，不要编号，不要解释。
最多输出 3 个实体。

用户问题：{query}
"""
    try:
        response = call_llm(prompt)
        entities = [line.strip() for line in response.strip().split('\n') 
                   if line.strip() and not line.strip().startswith('#')]
        return entities[:3]
    except Exception as e:
        print(f'实体提取失败: {e}')
        return []


# ==================== 图谱可视化 ====================

@app.route('/visualize', methods=['GET'])
def visualize():
    global kg_graph
    
    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        return jsonify({'error': '图谱为空，请先构建'}), 400
    
    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(kg_graph, k=3, iterations=50, seed=42)
    
    node_sizes = [500 + kg_graph.degree(node) * 200 for node in kg_graph.nodes()]
    node_colors = ['#FFA500' if kg_graph.degree(node) > 1 else '#ADD8E6' for node in kg_graph.nodes()]
    
    nx.draw_networkx_nodes(kg_graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(kg_graph, pos, edge_color='#888888', width=1.5, alpha=0.5, arrows=True, arrowsize=15)
    
    labels = {node: node for node in kg_graph.nodes()}
    nx.draw_networkx_labels(kg_graph, pos, labels, font_size=7, font_weight='bold')
    
    edge_labels = {(u, v): d.get('relation', '') for u, v, d in kg_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(kg_graph, pos, edge_labels, font_size=5)
    
    plt.title(f'Knowledge Graph\n{kg_graph.number_of_nodes()} Entities, {kg_graph.number_of_edges()} Relations', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    img_path = os.path.join(STATIC_DIR, 'knowledge_graph.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'📊 图谱可视化已保存: {img_path}')
    
    return jsonify({
        'status': 'ok',
        'node_count': kg_graph.number_of_nodes(),
        'edge_count': kg_graph.number_of_edges(),
        'image_url': 'http://127.0.0.1:5000/static/knowledge_graph.png'
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)


# ==================== 图谱统计 ====================

@app.route('/stats', methods=['GET'])
def stats():
    global kg_graph, triplet_store
    
    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        return jsonify({'error': '图谱为空'}), 400
    
    important_entities = sorted(kg_graph.out_degree(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'node_count': kg_graph.number_of_nodes(),
        'edge_count': kg_graph.number_of_edges(),
        'triplet_count': len(triplet_store),
        'important_entities': [{'entity': e, 'connections': d} for e, d in important_entities]
    })


# ==================== 启动服务 ====================

if __name__ == '__main__':
    load_graph()
    print('🧠 GraphRAG 知识图谱服务启动中...')
    print('   端口: 5000')
    print('   可用接口:')
    print('     POST /build_graph  - 构建知识图谱')
    print('     POST /query        - 图谱查询')
    print('     GET  /visualize    - 生成可视化图片')
    print('     GET  /stats        - 图谱统计')
    app.run(host='127.0.0.1', port=5000, debug=False)