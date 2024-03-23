<template>
  <q-item :to="{name:routeName}" clickable v-ripple>
        <q-item-section>
          <q-item-label overline>Overline Here</q-item-label>
          <q-item-label>{{routeName}}</q-item-label>
          <q-item-label caption>Caption Here</q-item-label>
        </q-item-section>
      </q-item>
</template>

<script setup lang="ts">
defineOptions({
  routeName: 'PageRoute'
});

export interface PageRoute {
  routeName: string
};

withDefaults(defineProps<PageRoute>(), {
  routeName: ''
});
</script>
