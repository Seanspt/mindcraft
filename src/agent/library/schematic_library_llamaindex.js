import { ChromaVectorStore } from "@llamaindex/chroma";
import { Ollama } from "@llamaindex/ollama";

import {
    VectorStoreIndex,
    Settings,
    getResponseSynthesizer,
    PromptTemplate,
} from "llamaindex";
import { HuggingFaceEmbedding } from "@llamaindex/huggingface";
import { env } from '@huggingface/transformers';
env.remoteHost = "https://hf-mirror.com";


Settings.embedModel = new HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2");

export class SchematicLibrary {
    static #instance;
    constructor() {
        if (SchematicLibrary.#instance) {
            return SchematicLibrary.#instance;
        }else{
            SchematicLibrary.#instance = this;
            this.queryEngine = null;
            this.llm = null;
        }
    }

    static getInstance() {
        if (!SchematicLibrary.#instance) {
            SchematicLibrary.#instance = new SchematicLibrary();
        }
        return SchematicLibrary.#instance;
    }

    async query(agent, query) {
        if (!this.queryEngine){
              const ollama = new Ollama({
                    model: "qwen2.5", 
                });
            Settings.llm = ollama
            const vectorStore = new ChromaVectorStore({ collectionName: "minecraft_schematics_llamaindex_default_embed" }); 
            const index = await VectorStoreIndex.fromVectorStore(vectorStore);
            const newTextQaPrompt = new PromptTemplate({
                                templateVars: ["context", "query"],
                                template: `你是一个我的世界建筑查询助手，用户通过查询语句来搜索数据库中感兴趣的建筑蓝图。
                                请根据用户的查询语句，分别介绍查询得到的蓝图的基本信息及与查询相关的特点。
                                查询得到的建筑描述如下：
                                ---------------------
                                {context}
                                ---------------------
                                用户的查询是: {query}
                                请给出你的介绍:`,
                                });

            this.queryEngine = index.asQueryEngine({ 
                similarityTopK: 1, 
            });
            this.queryEngine.updatePrompts({
                "responseSynthesizer:textQATemplate": newTextQaPrompt,
            });
        }     
        console.log("start query");
        const { message, sourceNodes } = await this.queryEngine.query({
            query: query,
        });

        sourceNodes.forEach((sourceNode, index) => {
            console.log(`来源 ${index + 1}: (相关性得分: ${sourceNode.score?.toFixed(4)})`);
            
            // 访问元数据对象
            const metadata = sourceNode.node.metadata;
            
            // 打印你感兴趣的元数据字段
            console.log(`  - ID: ${metadata.id}`);
            console.log(`  - 标题: ${metadata.title}`);
            console.log(`  - 分类: ${metadata.category}`);
            console.log(`  - 主题: ${metadata.theme}`);
            console.log(`  - 图片链接: ${metadata.image_urls}`);
            console.log(`  - 来源网址: ${metadata.source_url}`);
            // 你也可以打印原始的文本内容，用于调试
            // console.log(`  - 原始描述: "${sourceNode.node.getText().substring(0, 100)}..."`);
            console.log("\n");
            });
        console.log("query end");
        // console.log(message.content);
        return message.content;
    }
}
