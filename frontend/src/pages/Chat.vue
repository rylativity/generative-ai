<template>
  <q-page padding>
    <h3>
      Chat Page Content Here
    </h3>
    <p> {{ refs }}</p>
    <h3 
      v-for="(message, i) in messages"
      :key="i"
      >
      {{ message }}
    </h3>
    <q-btn
      @click="sendMessage('Hi')">

    </q-btn>
  </q-page>
</template>

<script setup lang="ts">
import { generate } from 'src/utils/inference';
import { reactive } from 'vue';
import { storeToRefs } from 'pinia';

import { useGenerationParamsStore } from 'src/stores/store';


const generationParamsStore = useGenerationParamsStore()

const refs = storeToRefs(generationParamsStore);

defineOptions({
  name: 'ChatPage'
});



const messages: string[] = reactive([])
async function sendMessage(message:string) {
  let response = await generate(message)
  messages.push(response.text)
}

</script>
