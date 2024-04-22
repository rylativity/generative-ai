<template>
  <div>
    <input 
    v-model="promptTemplate"
    type="text">
    <q-separator/>
    <p>Output</p>
    <p>{{ promptTemplateFields }}</p>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';

// import { generate } from 'src/utils/inference';
// import { CHATPROMPTTEMPLATE } from 'src/utils/promptTemplates';
// import { reactive, ref } from 'vue'
// import { Notify } from 'quasar'

var promptTemplate = ref('')

const promptTemplateFields = computed(() => {
  return promptTemplate
})

</script>
