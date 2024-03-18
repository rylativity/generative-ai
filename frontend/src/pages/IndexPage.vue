<template>
  <q-page class="row items-center justify-evenly">

    <h3>
      INDEX PAGE CONTENT
    </h3>

    <p>{{ refs }}</p>
  </q-page>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia';

import { useGenerationParamsStore } from 'src/stores/store';

defineOptions({
  name: 'IndexPage'
});

const generationParamsStore = useGenerationParamsStore()

const refs = storeToRefs(generationParamsStore);
</script>
