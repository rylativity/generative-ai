<template>
  <q-page class="row items-center justify-evenly">

    <h3>
      Setup/Config
    </h3>
    <ContentManager/>


  </q-page>
</template>

<script setup lang="ts">
import ContentManager from 'src/components/ContentManager.vue';

defineOptions({
  name: 'IndexPage'
});

</script>