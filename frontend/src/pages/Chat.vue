<template>
  <q-page padding>
    <chat-interface/>
  </q-page>
</template>

<script setup lang="ts">
import ChatInterface from 'src/components/ChatInterface.vue'

defineOptions({
  name: 'ChatPage'
});

</script>
