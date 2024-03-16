<template>
    <div class="q-pa-md" style="max-width: 400px">
  
      <!-- <q-form
        @submit="onSubmit"
        @reset="onReset"
        class="q-gutter-md"
      > -->
      <q-form
        class="q-gutter-md"
      >
      <div class="q-pa-md">
        <q-toggle
          v-model="doSample"
          color="primary"
          keep-color
        />
      </div>
        
  
        <q-input
          filled
          type="number"
          v-model="minNewTokens"
          label="Your age *"
          lazy-rules
          :rules="[
            val => val !== null && val !== '' || 'Please type your age',
            val => val > 0 && val < 100 || 'Please type a real age'
          ]"
        />
  
        <!-- <q-toggle v-model="accept" label="I accept the license and terms" /> -->
  
        <div>
          <q-btn label="Submit" type="submit" color="primary"/>
          <q-btn label="Reset" type="reset" color="primary" flat class="q-ml-sm" />
        </div>
      </q-form>
  
    </div>
  </template>
  
<script setup lang="ts">
// import { computed, ref } from 'vue';
// import { GenerationParams } from 'components/models';
// import { reactive } from 'vue';
import { storeToRefs } from 'pinia';
import { generationParamsStore } from 'src/stores/store';

const store = generationParamsStore();

const {
  doSample,
  minNewTokens,
  // maxNewTokens,
  // repetitionPenalty,
  // temperature,
  // topP,
  // topK,
} = storeToRefs(store);
// const params = reactive(<GenerationParams>({
//   doSample:true,
//   minNewTokens:1,
//   maxNewTokens:25,
//   repetitionPenalty:1.0,
//   temperature:0.7,
//   topP:1.0,
//   topK:50,
// }))

// const todoCount = computed(() => props.todos.length);
</script>