import { NEXT_GEN_OPERATION, POST_FORMAT, RESET_OPERATION, SAVE_IMAGES_OPERATION } from 'Constants';
import { API_URL } from 'Constants';

// helper function for post request
export function post(url, data) {
    return fetch(url, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data) });
}

function receivePostReponse(response) {
    if (response.status === 200) {
        // got new population
        return Promise.resolve(response.json())
    }
    else if (response.status === 502) {
        // got gateway error
        return Promise.reject(response.json())
    }
    else {

        // failed, reject promise
        return Promise.reject(response.status);
    }
}

/* gets a new population from the server, given the current population
 and the configuration */
export function nextGeneration(currentPopulation, config, clipText, failedToast) {

    // ensure that at least one individual is selected
    let at_least_one_selected = false;
    for (let i = 0; i < currentPopulation.length; i++) {
        if (currentPopulation[i].selected) {
            at_least_one_selected = true;
            break;
        }
    }
    if (!at_least_one_selected) {
        failedToast("No individuals selected")

        // don't send empty request
        return Promise.reject("No individuals selected");
    }

    // clone the population to leave the original unchanged
    const population = JSON.parse(JSON.stringify(currentPopulation));

    // clear out images, don't send to server
    for (let i = 0; i < population.length; i++) {
        population[i].image = undefined
    }

    // construct request
    const postData = POST_FORMAT
    postData.operation = NEXT_GEN_OPERATION
    postData.population = population
    postData.config = config
    postData.clip_text = clipText

    // send request
    return new Promise((resolve, reject) => {
        post(API_URL, postData).then((response) => {
            resolve(receivePostReponse(response))
        }
        ).catch((error) => {
            console.log("Error: " + error);
            reject(error);
        })
    })
}
/* gets population images from the server for saving, given the current population
 and the configuration */
export function saveIndividuals(currentPopulation, config) {

    // ensure that at least one individual is selected
    let at_least_one_selected = false;
    for (let i = 0; i < currentPopulation.length; i++) {
        if (currentPopulation[i].selected) {
            at_least_one_selected = true;
            break;
        }
    }
    if (!at_least_one_selected) {
        // don't send empty request
        return Promise.reject("No individuals selected");
    }

    // clone the population to leave the original unchanged
    const population = JSON.parse(JSON.stringify(currentPopulation));

    // clear out images, don't send to server
    for (let i = 0; i < population.length; i++) {
        population[i].image = undefined
    }

    // construct request
    const postData = POST_FORMAT
    postData.operation = SAVE_IMAGES_OPERATION
    postData.population = population
    postData.config = config
    postData.clip_text = ""

    // send request
    return new Promise((resolve, reject) => {
        post(API_URL, postData).then((response) => {
            resolve(receivePostReponse(response))
        }
        ).catch((error) => {
            console.log("Error: " + error);
            reject(error);
        })
    })
}

// gets a new population from the server, given the configuration
export function initialPopulation(config) {
    // construct request
    let postData = POST_FORMAT
    postData.operation = RESET_OPERATION
    postData.config = config
    postData.clip_text = ""

    // send request
    return new Promise((resolve, reject) => {
        post(API_URL, postData).then((response) => {
            if (response.status === 200) {
                resolve(response.json())
            }
            else {
                reject(response.status);
            }
        }
        ).catch((error) => {
            console.log("Error: " + error);
            reject(error);
        })
    })
}

export function getImageUrl(individual) {
    // create image from individual
    let parsed = individual.image
    if(parsed === null) {
        console.log(individual.image)
        parsed = individual.image
    }
    // create url from base64 string
    const url = "data:image/jpg;base64," + parsed
    return url
}