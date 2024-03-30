<template>
  <div class="q-pa-md row justify-center">
    <div style="width: 100%; max-width: 800px">
      <q-chat-message
        v-for="(message, i) in messages"
        :key="i"
        :name="message.name"
        :avatar="message.avatar"
        :text="message.text"
        :sent="message.sent"
      />
      <q-chat-message 
      v-if="generatingResponse"
      name="Assistant"
      avatar="https://cdn.quasar.dev/img/avatar2.jpg"
      :sent="false"
      >
       <q-spinner-dots size="2rem" />
      </q-chat-message>
      
      <q-input v-model="chatInput" @keydown.enter.prevent="sendMessage(chatInput)" outlined  label="Send a message">
        <template v-slot:after>
          <q-btn 
          @click="sendMessage(chatInput)"
          round dense flat icon="send" />
        </template>
      </q-input>
      
    </div>
  </div>
</template>

<script setup lang="ts">
import { generate } from 'src/utils/inference';
import { CHATPROMPTTEMPLATE } from 'src/utils/promptTemplates';
import { reactive, ref } from 'vue'
import { Notify } from 'quasar'


type ChatMessage = {
  name:string,
  avatar:string, 
  text:string[], 
  sent:boolean
}

var chatInput=''
var messages: ChatMessage[] = reactive([])

function createUserMessage(message:string):ChatMessage {
  return {
    name:'User',
    avatar:'https://cdn.quasar.dev/img/avatar1.jpg',
    text:[message],
    sent:true
  }
}

var generatingResponse = ref()

function createMessageString(chatMessage:ChatMessage):string {
  return `${chatMessage.name}: ${chatMessage.text[0]}`
}

function createChatPrompt(chatMessages:ChatMessage[], template=CHATPROMPTTEMPLATE){
  let messagesHistoryString:string
  if (!(chatMessages.length > 1)) {
    messagesHistoryString = createMessageString(chatMessages[0])
  } else {
    messagesHistoryString = chatMessages.map((message) => {
      return createMessageString(message)
    }).join('\n\n')
  }

  return template(messagesHistoryString)
}

async function sendMessage(message:string) {

  if (message === '') {
    Notify.create({
        color:'negative',
        position:'top',
        message:'Enter a message',
        icon: 'report_problem'
    })
    return
  }

  generatingResponse.value = true

  const userMessage = createUserMessage(message)
  messages.push(userMessage)

  const chatPrompt = createChatPrompt(messages)
  chatInput=''

  console.log(chatPrompt)
  let response = await generate(chatPrompt)
  
  const responseMessage: ChatMessage = {
    name:'Assistant',
    avatar:'https://cdn.quasar.dev/img/avatar2.jpg',
    text:[response.text],
    sent:false
  }

  generatingResponse.value = false
  messages.push(responseMessage)
}

</script>
