import { RouteRecordRaw } from 'vue-router';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'root',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      { path: '', name:'Home', component: () => import('pages/IndexPage.vue') },
      { path: 'basic-prompting/', name:'Basic Prompting', component: () => import('pages/BasicPrompting.vue')},
      { path: 'chat/', name:'Chat', component: () => import('pages/Chat.vue')},
    ],
  },
  // Always leave this as last one,
  // but you can also remove it
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/ErrorNotFound.vue'),
  },
];

export default routes;
