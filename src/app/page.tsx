import { Header } from "@/components/header"
import { Chat } from "./Chat"

export default async function ChatGPTWithX() {

    return (
        <>  
            <Header
                heading="ChatGPT with X"
                subheading="A ChatGPT alternative that chats with documents."
            />
            <Chat />
        </>
    )
}