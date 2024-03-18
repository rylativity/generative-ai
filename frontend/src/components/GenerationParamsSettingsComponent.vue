<template>
    <div class="q-pa-md" style="max-width: 400px">
  
      <q-form
        class="q-gutter-md"
      >
      <div class="q-pa-md">
        <q-toggle
          v-model="doSample"
          :label="doSample ? 'Sampling Decoding' : 'Greedy Decoding'"
          color="primary"
          keep-color
        />

        <q-input
          filled
          type="number"
          v-model="minNewTokens"
          label="Min New Tokens"
          lazy-rules
          :rules="[
            val => val !== null || 'Cannot be empty',
            val => val > 0 && val <= 4096 || 'Must be between 1 and 4096'
          ]"
        />

        <q-input
          filled
          type="number"
          v-model="maxNewTokens"
          label="Max New Tokens"
          lazy-rules
          :rules="[
            val => val !== null || 'Cannot be empty',
            val => val > 0 && val <= 4096 || 'Must be between 1 and 4096'
          ]"
        />

        <q-input
          filled
          type="number"
          v-model.number="repetitionPenalty"
          label="Repetition Penalty"
          lazy-rules
          :rules="[
            val => val !== null && val !== '' || 'Cannot be empty',
            val => val >= 1.0 && val <= 2.0 || 'Must be between 1.0 and 2.0'
          ]"
        />

        <q-input
          filled
          type="number"
          v-model="topP"
          label="Top P"
          lazy-rules
          :rules="[
            val => val !== null && val !== '' || 'Cannot be empty',
            val => val >= 0.01 && val <= 1.00 || 'Must be between 0.01 and 1.00'
          ]"
        />

        <q-input
          filled
          type="number"
          v-model="topK"
          label="Top K"
          lazy-rules
          :rules="[
            val => val !== null && val !== '' || 'Cannot be empty',
            val => val >= 1 && val <= 100 || 'Must be between 1 and 100'
          ]"
        />

        <q-input
          filled
          type="number"
          v-model="temperature"
          label="Temperature"
          lazy-rules
          :rules="[
            val => val !== null && val !== '' || 'Cannot be empty',
            val => val >= 0.01 && val <= 1.50 || 'Must be between 0.01 and 1.50'
          ]"
        />
      </div>
      </q-form>
  
    </div>
  </template>
  
<script setup lang="ts">
// import { computed, ref } from 'vue';
// import { GenerationParams } from 'components/models';
// import { reactive } from 'vue';
import { storeToRefs } from 'pinia';
import { useGenerationParamsStore } from 'src/stores/store';

const store = useGenerationParamsStore();

const {
  doSample,
  minNewTokens,
  maxNewTokens,
  repetitionPenalty,
  temperature,
  topP,
  topK,
} = storeToRefs(store);
</script>
