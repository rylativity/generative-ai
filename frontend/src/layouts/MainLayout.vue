<template>
  <q-layout view="lHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>
        <q-btn
          flat
          dense
          round
          icon="menu"
          aria-label="Menu"
          @click="toggleLeftDrawer"
        />

        <q-toolbar-title>
          Generative AI
        </q-toolbar-title>

        <a href="http://localhost:5601"
          target="_blank">
          <q-btn 
          >
          <q-tooltip>
            <span>
              Logging Dashboard
            </span>
          </q-tooltip>
          <q-icon size="xs" name="dashboard" />
        </q-btn>
        </a>

        &nbsp;&nbsp;

        <q-btn @click="displayGenerationParams = !displayGenerationParams">
          <q-tooltip>
            Generation Parameter Settings
          </q-tooltip>
          <q-icon size="xs" name="settings_input_component" />
        </q-btn>

        &nbsp;&nbsp;

        <q-btn 
          @click="checkApiStatus(true)"
          :color="apiStatus ? 'green' : 'red'"
          >
          <q-tooltip>
            <span v-if=apiStatus>
              Inference API Connected
            </span>
            <span v-else>
              Inference API Disconnected
            </span>
            <span>
              (last checked {{ lastChecked }})
            </span>
          </q-tooltip>
          <q-icon size="xs" :name="apiStatus ? 'check_circle' : 'build_circle'" />
        </q-btn>
        
      </q-toolbar>
      <span>
        Generation Params Settings:
        {{ generationParamsRefs }}
      </span>
    </q-header>

    <q-drawer
      v-model="leftDrawerOpen"
      show-if-above
      bordered
    >
      <q-list>
        <q-item-label
          header
        >
          Application Interfaces
        </q-item-label>

        <!-- <RouterLink 
          v-for="route in routeNames" 
          :key="route"
          :to="{name:route}">
          {{ route }}
        </RouterLink> -->

        <PageRoute 
          v-for="route in routeNames"
          :key="route"
          :route-name="route">
        </PageRoute>



        <!-- <EssentialLink
          v-for="link in linksList"
          :key="link.title"
          v-bind="link"
        /> -->
      </q-list>
    </q-drawer>



    <q-btn @click="displayGenerationParams = !displayGenerationParams">
        Close
    </q-btn>
    <q-drawer 
      side="right" 
      v-model="displayGenerationParams">
      <GenerationParamsSettings>

      </GenerationParamsSettings>
    </q-drawer>

    <q-page-container>
      <router-view />
    </q-page-container>
  </q-layout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import GenerationParamsSettings from 'src/components/GenerationParamsSettingsComponent.vue';
import routes from 'src/router/routes';
import PageRoute from 'src/components/PageRoute.vue';
import { useGenerationParamsStore } from 'src/stores/store';
import { storeToRefs } from 'pinia';
import { healthcheck } from 'src/utils/inference';

defineOptions({
  name: 'MainLayout'
});

const generationParamsStore = useGenerationParamsStore()

const generationParamsRefs = storeToRefs(generationParamsStore)

const displayGenerationParams = ref(false)

const rootRoute = routes.find(route => route.name === 'root')
const routeNames = rootRoute?.children?.map( (route) => {
  return route.name?.toString()
})

console.log(routeNames)

const leftDrawerOpen = ref(false);

function toggleLeftDrawer () {
  leftDrawerOpen.value = !leftDrawerOpen.value;
}

var apiStatus = ref(true)
var lastChecked = ref('Never')
async function checkApiStatus(notification=false) {
  apiStatus.value = await healthcheck(notification=notification)
  lastChecked.value = Date().toString()
}

setInterval(() => {
  checkApiStatus()
}, 10000)
</script>

<style scoped>
q-select {
  max-width: 1%;
}
</style>