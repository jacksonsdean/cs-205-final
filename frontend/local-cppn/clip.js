// import { pipeline, CLIPTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
// import { env, AutoTokenizer, CLIPTextModelWithProjection } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
//import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { AutoTokenizer, CLIPTextModelWithProjection, AutoProcessor, CLIPVisionModelWithProjection, RawImage, cos_sim } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';

import { env} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0';
env.allowLocalModels = false; // Skip local model check


export async function computeSimilarities(prompt, images){

    // Load tokenizer and text model
    const tokenizer = await AutoTokenizer.from_pretrained('jinaai/jina-clip-v1');
    const text_model = await CLIPTextModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    
    // Load processor and vision model
    const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
    const vision_model = await CLIPVisionModelWithProjection.from_pretrained('jinaai/jina-clip-v1');
    
    // Run tokenization
    const texts = [prompt];
    const text_inputs = tokenizer(texts, { padding: true, truncation: true });
    
    // Compute text embeddings
    const { text_embeds } = await text_model(text_inputs);
    
    const raw_images = images.map(image => new RawImage(image, 256, 256, 3));
    console.log(raw_images);
    const image_inputs = await processor(raw_images);
    
    // Compute vision embeddings
    const { image_embeds } = await vision_model(image_inputs);
    
    //  Compute similarities
    console.log(cos_sim(text_embeds[0].data, image_embeds[0].data)) // text-image cross-modal similarity

    // return cos_sim(text_embeds[0].data, image_embeds[0].data)  // TODO: all the images   
    // map
    let result = [];
    for (let i = 0; i < cos_sim(text_embeds[0].data, image_embeds[0].data).length; i++) {
        result.push(cos_sim(text_embeds[0].data, image_embeds[0].data)[i]);
    }
    return result;
}
