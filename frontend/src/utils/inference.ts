import { AxiosError, AxiosResponse } from 'axios'
import { api } from 'boot/axios'
import { Notify } from 'quasar'
import { useGenerationParamsStore } from 'src/stores/store'
import { storeToRefs
 } from 'pinia'
const generationParamsStore = useGenerationParamsStore()


export async function generate(input:string,){
    const refs = storeToRefs(generationParamsStore);

    const generationParams = {
        'do_sample':refs.doSample.value,
        'min_new_tokens':refs.minNewTokens.value,
        'max_new_tokens':refs.maxNewTokens.value,
        'repetition_penalty':refs.repetitionPenalty.value,
        'temperature':refs.temperature.value,
        'top_p':refs.topP.value,
        'top_k':refs.topK.value,
        }
    console.log('generating')
    console.log(generationParams)
    const payload = {...generationParams, 'input':input}
    console.log(payload)
    const response = await api.post(
        '/api/generate', payload
        )
        .then( (response:AxiosResponse) => {
            Notify.create({
                color:'positive',
                position:'top',
                message:'Inference Succeeded',
                icon: 'success'
            })
            return response.data
        })
        .catch( (error: Error | AxiosError) => {
            console.log(error)
            Notify.create({
                color:'negative',
                position:'top',
                message:'Inference Failed',
                icon: 'report_problem'
            })
        })
    return response
}