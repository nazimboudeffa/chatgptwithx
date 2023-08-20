import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";

import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export async function promptChatGPT( apiKey : string, prompt: string) {

  const loader = new DirectoryLoader("./src/data", {
    ".txt": (path) => new TextLoader(path),
  });

  console.log("Loading docs...");
  const docs = await loader.load();
  console.log("Loaded docs:", docs.length);

  const VECTOR_STORE_PATH = "./src/data-index";

  function normalizeDocs(docs : any) {
      return docs.map((doc : any) => {
        if (typeof doc.pageContent === "string") {
          return doc.pageContent;
        } else if (Array.isArray(doc.pageContent)) {
          return doc.pageContent.join("\n");
        }
      });
    }

  const model = new OpenAI({ openAIApiKey: apiKey });

  const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
  const normalizedDocs = normalizeDocs(docs);
  const splitDocs = await textSplitter.createDocuments(normalizedDocs)
  
  let vectorStore = await HNSWLib.fromDocuments(
      splitDocs,
      new OpenAIEmbeddings( { openAIApiKey: apiKey } )
  );

  await vectorStore.save(VECTOR_STORE_PATH);

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

  console.log("Creating retrieval chain...");
  const result = await chain.call({ query: prompt });
  console.log("Result:", result);
  return result
}