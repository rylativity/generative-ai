import { defineStore } from 'pinia';

export const useGenerationParamsStore = defineStore('generationParams', {
  state: () => ({
      doSample:true,
      minNewTokens:1,
      maxNewTokens:100,
      repetitionPenalty:1.0,
      temperature:0.7,
      topP:1.0,
      topK:50,
  }),

  getters: {
    // doubleCount (state) {
    //   return state.counter * 2;
    // }
  },

  actions: {
    // increment () {
    //   this.counter++;
    // }
  }
});
