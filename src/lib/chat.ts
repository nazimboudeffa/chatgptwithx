import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";

import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import fs from "fs";

const loader = new DirectoryLoader("./data", {
    ".txt": (path) => new TextLoader(path),
  });


const docs = loader.load();

function normalizeDocs(docs : any) {
    return docs.map((doc : any) => {
      if (typeof doc.pageContent === "string") {
        return doc.pageContent;
      } else if (Array.isArray(doc.pageContent)) {
        return doc.pageContent.join("\n");
      }
    });
  }

const model = new OpenAI({});

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });
const normalizedDocs = normalizeDocs(docs);

let vectorStore = HNSWLib.fromDocuments(
    textSplitter.createDocuments(normalizedDocs),
    new OpenAIEmbeddings()
);

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

export async function promptChatGPT(apiKey: string, prompt: string) {

    const result = await chain.call({ query: prompt });

    return result ? { text: result } : null
}