<template>
  <q-page padding>
    <h3>
      Chat Page Content Here
    </h3>
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
import { useGenerationParamsStore } from 'src/stores/store';

import { storeToRefs } from 'pinia';
import { reactive } from 'vue';

defineOptions({
  name: 'ChatPage'
});

const generationParamsStore = useGenerationParamsStore()
const refs = storeToRefs(generationParamsStore);
const params = {
      'input':'',
      'doSample':refs.doSample.value,
      'minNewTokens':refs.minNewTokens.value,
      'maxNewTokens':refs.maxNewTokens.value,
      'repetitionPenalty':refs.repetitionPenalty.value,
      'temperature':refs.temperature.value,
      'topP':refs.topP.value,
      'topK':refs.topK.value,
      }

const messages: string[] = reactive([])
async function sendMessage(message:string) {
  let inputObject = params
  inputObject.input=message
  let response = await generate(inputObject)
  messages.push(response.text)
}

</script>
