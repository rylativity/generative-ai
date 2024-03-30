const CHATPROMPTTEMPLATE = (messages:string) => {
    return `<s>[INST]
    <<SYS>>
    You are a conversational assistant. Answer honestly and if you do not have enough information to answer, say so. 
    <</SYS>>
    
    ${messages}[/INST]
    
    Assistant:
    `
}

export { CHATPROMPTTEMPLATE }