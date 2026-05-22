import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = 'http://127.0.0.1:8787';

// 读取你的20个测试问题
const testData = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../test_questions.json'), 'utf-8')
);
const testQuestions = testData.test_questions;

async function collectEvalData() {
  console.log(`📊 开始收集评估数据，共 ${testQuestions.length} 个问题\n`);
  
  // 提取所有问题文本
  const queries = testQuestions.map(q => q.question);
  
  try {
    const response = await fetch(`${BASE_URL}/api/collect-eval-data`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ queries })
    });
    
    const result = await response.json();
    console.log(`✅ ${result.message}`);
    console.log(`📁 数据已保存至项目根目录的 eval_data.json`);
  } catch (error) {
    console.error('❌ 收集失败:', error.message);
    console.log('请确保 Node.js 服务已启动在 http://localhost:8787');
  }
}

collectEvalData();