<template>
  <div class="q-pa-md row justify-center">
    <div style="width: 100%; max-width: 400px">
      <q-chat-message
        v-for="(message, i) in messages"
        :key="i"
        :name="message.name"
        :avatar="message.avatar"
        :text="message.text"
        :sent="message.sent"
      />
      <q-chat-message
        name="Jane"
        avatar="https://cdn.quasar.dev/img/avatar2.jpg"
        :text="['doing fine, how r you?']"
      />
      <q-btn
      @click="sendMessage('Hi')">

    </q-btn>
    </div>
  </div>
</template>

<script setup lang="ts">
import { generate } from 'src/utils/inference';
import { reactive } from 'vue'

type ChatMessage = {
  name:string,
  avatar:string, 
  text:string[], 
  sent:boolean
}

var messages: ChatMessage[] = reactive([])

function createUserMessage(message:string):ChatMessage {
  return {
    name:'User',
    avatar:'https://cdn.quasar.dev/img/avatar1.jpg',
    text:[message],
    sent:true
  }
}

async function sendMessage(message:string) {

  const userMessage = createUserMessage(message)
  messages.push(userMessage)
  
  let response = await generate(message)
  
  const responseMessage: ChatMessage = {
    name:'Assistant',
    avatar:'https://cdn.quasar.dev/img/avatar2.jpg',
    text:[response.text],
    sent:false
  }

  messages.push(responseMessage)
}


console.log(messages)

</script>
