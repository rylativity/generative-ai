<template>
  <q-page padding>
    <PromptBuilder/>
  </q-page>
</template>

<script setup lang="ts">
import PromptBuilder from 'components/PromptBuilder.vue'

defineOptions({
  name: 'BasicPromptingPage'
});
</script>
