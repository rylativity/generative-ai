<template>
  <q-page class="row items-center justify-evenly">

    <generation-params-settings-component>
      
    </generation-params-settings-component>

    <p>{{ refs }}</p>
  </q-page>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia';

import { useGenerationParamsStore } from 'src/stores/store';
import GenerationParamsSettingsComponent from 'components/GenerationParamsSettingsComponent.vue';

defineOptions({
  name: 'IndexPage'
});

const generationParamsStore = useGenerationParamsStore()

const refs = storeToRefs(generationParamsStore);
</script>
