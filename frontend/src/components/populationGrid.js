import React from 'react';
import { API_URL, INITIAL_REQUEST } from '../Constants';
import { post } from '../util';

class IndividualButton extends React.Component {
    render() {
        return (
            <button className="individual-button" onClick={this.props.onClick}>
                {/* <img src={this.props.image} alt={this.props.alt} /> */}
                {this.props.name}
                {this.props.image}
            </button>
        )
    }
}

class PopulationGrid extends React.Component {

    constructor(props) {
        super(props);
        this.state = { population: [] };
    }

    componentDidMount() {
        const apiUrl = API_URL;
        console.log(process.env.REACT_APP_NODE_ENV);
        console.log(apiUrl)
        post(apiUrl, INITIAL_REQUEST)
            .then((response) => {
                console.log(response)
                if (response.status === 200 ){
                    return response.json()
                }
                else{
                    console.log("Error: " + response.status);
                    return Promise.reject(response.status);
                }
            }
            ).catch((error) => {
                console.log("Error: " + error);})
                .then((data) => {
                    if ("body" in data && typeof data.body !== "object") {
                    data =JSON.parse(data["body"]);
                }
                console.log(data);
                const pop = data["population"];
                this.setState({ population: pop });
            })
            .catch((error) => {
                console.log("Error: " + error);})
    }

    render() {
        if (typeof(this.state.population) === 'undefined' || this.state.population.length === 0) {
            return (<p>Please wait...</p>)
        }
        // list of all the population ids
        return (<div>
            <ul>
                {this.state.population.map(obj, index=> <li key={index}> <IndividualButton name={index} image={i["image"]}></IndividualButton> </li>)}
            </ul>
        </div>
        );
    }
}
export default PopulationGrid;