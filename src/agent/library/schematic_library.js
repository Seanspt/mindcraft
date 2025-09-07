import { ChromaClient } from 'chromadb';
import { pipeline } from '@huggingface/transformers';
import { env } from '@huggingface/transformers';
env.remoteHost = "https://hf-mirror.com";

// 自定义嵌入函数
class EmbeddingFunction {
  constructor(modelName) {
    this.modelName = modelName;
    this.extractor = null;
  }

  async generate(texts) {
    if (!this.extractor) {
      this.extractor = await pipeline('feature-extraction', this.modelName);
    }
    const output = await this.extractor(texts, { pooling: 'mean', normalize: true });
    return output.tolist();
  }
}

export class SchematicLibrary {
    static #instance;
    constructor() {
        if (SchematicLibrary.#instance) {
            return SchematicLibrary.#instance;
        }else{
            SchematicLibrary.#instance = this;
            this.collection = null;
        }
    }

    static getInstance() {
        if (!SchematicLibrary.#instance) {
            SchematicLibrary.#instance = new SchematicLibrary();
        }
        return SchematicLibrary.#instance;
    }

    async query(query) {
        if (!this.collection){
            const client = new ChromaClient();
            this.collection = await client.getCollection({ 
                name: "minecraft_schematics_sample",
                embeddingFunction: new EmbeddingFunction('sentence-transformers/all-MiniLM-L6-v2'),
             });
        }
        
        const results = await this.collection.query({
            queryTexts: [query],
            nResults: 3,
        }); 

        if (!results.ids || !results.ids[0] || results.ids[0].length === 0) {
            return JSON.stringify([], null, 2);
        }

        const ids = results.ids[0];
        const documents = results.documents[0];
        const metadatas = results.metadatas[0];

        const reorganizedResults = ids.map((id, index) => ({
            id: id,
            document: documents[index],
            title: metadatas[index]?.title || null
        }));

        return JSON.stringify(reorganizedResults, null, 2);
    }
}
