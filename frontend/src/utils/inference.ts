import { AxiosError, AxiosResponse } from 'axios'
import { api } from 'boot/axios'
import { Notify } from 'quasar'

export async function generate({
    input='',
    doSample=true,
    minNewTokens=1,
    maxNewTokens=25,
    repetitionPenalty=1.0,
    temperature=0.7,
    topP=1.0,
    topK=50,
    }){
        
        const payload = {
            'input':input,
            'do_sample':doSample,
            'min_new_tokens':minNewTokens,
            'max_new_tokens':maxNewTokens,
            'repetition_penalty':repetitionPenalty,
            'temperature':temperature,
            'topP':topP,
            'topK':topK,
        }
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