import { Agent, Model, Doc, TLLMEvent } from '@smythos/sdk';
import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import readline from 'readline';
import inquirer from 'inquirer';

const __dirname$1 = process.cwd();
const BOOKS_NAMESPACE = "books";
const agent$1 = new Agent({
  id: "book-assistant",
  //<=== agent id is important for data isolation in vector DBs and Storage
  //the name of the agent, this is how the agent will identify itself
  name: "Book Assistant",
  //here we are using a builtin model
  //note that we are not passing an apiKey because we will rely on smyth vault for the model credentials
  model: "gpt-4o",
  //the behavior of the agent, this describes the personnality and behavior of the agent
  behavior: "You are a helpful assistant that can answer questions about the books."
});
const ramvec = agent$1.vectorDB.RAMVec(BOOKS_NAMESPACE, {
  embeddings: Model.OpenAI("text-embedding-3-small")
});
agent$1.addSkill({
  name: "index_book",
  description: "Use this skill to index a book in a vector database, the user will provide the path to the book",
  process: async ({ book_path }) => {
    const filePath = path.resolve(__dirname$1, book_path);
    if (!fs.existsSync(filePath)) {
      return `File resolved path to ${filePath} does not exist`;
    }
    const parsedDoc = await Doc.auto.parse(filePath);
    const name = path.basename(filePath);
    const result = await ramvec.insertDoc(name, parsedDoc);
    if (result) {
      return `Book ${name} indexed successfully`;
    } else {
      return `Book ${name} indexing failed`;
    }
  }
});
agent$1.addSkill({
  name: "lookup_book",
  description: "Use this skill to lookup a book in the vector database",
  process: async ({ user_query }) => {
    const result = await ramvec.search(user_query, {
      topK: 5
    });
    return result;
  }
});
const openlibraryLookupSkill = agent$1.addSkill({
  name: "get_book_info",
  description: "Use this skill to get information about a book",
  process: async ({ book_name }) => {
    const url = `https://openlibrary.org/search.json?q=${book_name}`;
    const response = await fetch(url);
    const data = await response.json();
    return data.docs[0];
  }
});
openlibraryLookupSkill.in({
  book_name: {
    description: "This need to be a name of a book, extract it from the user query"
  }
});

const __dirname = process.cwd();
const agentPath = path.resolve(__dirname, "./data/crypto-assistant.smyth");
const agent = Agent.import(agentPath, {
  id: "crypto-assistant",
  //<=== Chat persistence requires an explicitly identified agent
  model: "gpt-4o"
  //<=== we can override agent settings, here we are setting the model to gpt-4o
});

function runChat(chat) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: chalk.blue("You: ")
  });
  console.log(chalk.green(`
\u{1F680} ${chat.agentData.name} is ready!`));
  console.log(chalk.gray("Type your question below to talk to the agent."));
  console.log(chalk.gray('Type "exit" or "quit" to end the conversation.\n'));
  rl.on("line", (input) => handleUserInput(input, rl, chat));
  rl.on("close", () => {
    console.log(chalk.gray("Chat session ended."));
    process.exit(0);
  });
  rl.prompt();
}
async function handleUserInput(input, rl, chat) {
  if (input.toLowerCase().trim() === "exit" || input.toLowerCase().trim() === "quit") {
    console.log(chalk.green("\u{1F44B} Goodbye!"));
    rl.close();
    return;
  }
  if (input.trim() === "") {
    rl.prompt();
    return;
  }
  try {
    console.log(chalk.gray("Assistant is thinking..."));
    const streamChat = await chat.prompt(input).stream();
    process.stdout.write("\r");
    let first = true;
    streamChat.on(TLLMEvent.Content, (content) => {
      if (first) {
        content = chalk.green("\u{1F916} Assistant: ") + content;
        first = false;
      }
      process.stdout.write(chalk.white(content));
    });
    streamChat.on(TLLMEvent.End, () => {
      console.log("\n");
      rl.prompt();
    });
    streamChat.on(TLLMEvent.Error, (error) => {
      console.error(chalk.red("\u274C Error:", error));
      rl.prompt();
    });
    streamChat.on(TLLMEvent.ToolCall, (toolCall) => {
      console.log(
        chalk.yellow("[Calling Tool]"),
        toolCall?.tool?.name,
        chalk.gray(typeof toolCall?.tool?.arguments === "object" ? JSON.stringify(toolCall?.tool?.arguments) : toolCall?.tool?.arguments)
      );
    });
  } catch (error) {
    console.error(chalk.red("\u274C Error:", error));
    rl.prompt();
  }
}

const main = async () => {
  const { agentChoice } = await inquirer.prompt([
    {
      type: "list",
      name: "agentChoice",
      message: "Use arrow keys to select an agent and press enter to start the chat",
      choices: ["Book Assistant", "Crypto Assistant"]
    }
  ]);
  let agent$2;
  if (agentChoice === "Book Assistant") {
    agent$2 = agent$1;
  } else {
    agent$2 = agent;
  }
  const sessionId = `my-chat-session-${agentChoice.replace(" ", "-")}`;
  const chat = agent$2.chat({
    id: sessionId,
    persist: true
  });
  runChat(chat);
};
main();
//# sourceMappingURL=index.js.map
