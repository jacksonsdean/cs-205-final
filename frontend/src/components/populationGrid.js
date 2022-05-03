import React from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { DEFAULT_CONFIG, MAX_HISTORY } from '../Constants';
import { initialPopulation, nextGeneration, getImageUrl, saveIndividuals } from '../util';
import Grid from './Grid';
import styles from "./PopulationGrid.module.css";

class FailedToast extends React.Component {
    render() {
        return (
            <div>
                {this.props.text}
            </div>
        );
    }
}

class IndividualButton extends React.Component {

    constructor(props) {
        super(props)
        this.state = { individual: props.individual }
        this.clicked = this.clicked.bind(this)
    }

    clicked() {
        this.props.individual.selected = !this.props.individual.selected
        this.setState({ individual: this.props.individual })
    }

    render() {
        const individual = this.props.individual;
        const url = getImageUrl(individual);
        // choose the style based on whether the individual is selected
        const selectionStyle = individual.selected ? styles.selected : styles.unselected
        return (
            <button className={styles.individualButton + " " + selectionStyle} onClick={this.clicked} style={this.props.style}>
                <img className={styles.individualImg} src={url} alt={individual.name} />
            </button>
        )
    }
}

class LoadingSpinner extends React.Component {
    render() {
        return (
            <div className={styles.loadingSpinnerContainer}>
                <div className={styles.loadingSpinner} data-testid="spinner" />
            </div>
        );
    }
}

class NextGenerationButton extends React.Component {
    render() {
        return (
            <button style={this.props.style} className={styles.nextGenButton + " " + (this.props.loading ? styles.loading : "")} onClick={this.props.onClick} disabled={this.props.loading}>
                <div>
                    Next<br />
                    generation
                </div>
                <div>
                    {"\u25B6"}
                </div>
            </button>
        )
    }
}

class PreviousGenerationButton extends React.Component {
    render() {
        return (
            <button style={this.props.style} className={styles.nextGenButton + " " + (this.props.loading ? styles.loading : "")} onClick={this.props.onClick} disabled={this.props.loading}>
                <div>
                    Previous<br />generation
                </div>
                <div>
                    {"\u25C0"}
                </div>
            </button>
        )
    }
}
class SaveImagesButton extends React.Component {
    render() {
        return (
            <button style={this.props.style} className={styles.nextGenButton + " " + (this.props.loading ? styles.loading : "")} onClick={this.props.onClick} disabled={this.props.loading}>
                <div>
                    Save images
                </div>
                <div>
                    {"\u{0001f4be}"}
                </div>
            </button>
        )
    }
}

class PopulationGrid extends Grid {
    failedToast = (text) => toast(<FailedToast text={text} />, { type: "warning", autoClose: 4000 });

    constructor(props) {
        super(props);
        this.state = { population: [], loading: true };
        this.history = [];
        this.config = DEFAULT_CONFIG;

        // initialize seed to a random value
        this.config.seed = Math.round(Math.random() * 10000);

        // bind member functions
        this.nextGenerationClicked = this.nextGenerationClicked.bind(this);
        this.previousGenerationClicked = this.previousGenerationClicked.bind(this);
        this.saveImagesClicked = this.saveImagesClicked.bind(this);
    }

    /* Handles incoming population and image data from the server */
    handleNewData(data) {
        if ("body" in data && typeof data.body !== "object") {
            data = JSON.parse(data["body"]);
        }
        if (data.error) {
            console.log(data.error)
            return;
        }
        const current_pop = this.state.population;
        const next_pop = data["population"];
        if (next_pop === 'undefined' || next_pop.length === 0) {
            console.error("No next_population returned")
            return
        }

        for (let i = 0; i < current_pop.length; i++) {
            if (current_pop[i].selected) {
                // keep selected in population
                next_pop[i] = current_pop[i]
            }
            // deselect all
            next_pop[i].selected = false;
        }
        this.config.seed += 1; // increment seed
        this.setState({ population: next_pop, loading: false });
    }

    /* Handles the next generation button */
    nextGenerationClicked() {
        this.history.push(this.state.population) // put current population in history
        if (this.history.length > MAX_HISTORY) {
            this.history.shift() // remove oldest population from history
        }

        this.setState({ loading: true });
        nextGeneration(this.state.population, this.config, this.failedToast).then((data) => {
            this.handleNewData(data)
        }).catch((err) => {
            console.log(err)

            this.history.pop() // remove last population from history
            this.setState({ population: this.state.population, loading: false });
        })
    }

    /* Handles the previous generation button */
    previousGenerationClicked() {
        if (this.history.length === 0) {
            return
        }
        this.setState({ loading: false, population: this.history.pop() });
    }

    /* Handles the save images button */
    saveImagesClicked() {
        this.setState({ loading: true });
        saveIndividuals(this.state.population, this.config).then((data) => {
            if ("body" in data && typeof data.body !== "object") {
                data = JSON.parse(data["body"]);
            }
            if (data.error) {
                this.failedToast("Saving images failed. Try selecting fewer at once.")
                console.log(data.error)
                return;
            }
            const population = data["population"];
            for (let i = 0; i < population.length; i++) {
                const individual = population[i];
                if (individual.selected) {
                    continue
                }
                const url = getImageUrl(individual);
                // download image
                const link = document.createElement("a");
                link.href = url;
                link.download = "saved_genome_" + i.toString() + ".png";
                link.click();
            }
            this.setState({ loading: false });

        }).catch((err) => {
            console.log(err)
            this.failedToast("Saving images failed. Try selecting fewer at once.")

            this.setState({ population: this.state.population, loading: false });
        })
    }

    componentDidMount() {
        // get an initial population
        initialPopulation(this.config)
            .then((data) => {
                this.handleNewData(data)
            }).catch((err) => {
                console.log(err)
            })
    }

    render() {
        if (typeof (this.state.population) === 'undefined' || this.state.population.length === 0) {
            // no population yet, show loading spinner
            return <LoadingSpinner />
        }
        const gridWidth = this.config.res_w * (1 + Math.sqrt(this.config.population_size));
        const individualWidth = (100 / (1 + Math.sqrt(this.config.population_size))).toString() + "%";
        // a grid of the population's individuals' images as buttons
        return (
        <>
            <div className={styles.populationGrid} style={{ width: gridWidth, maxWidth: "95vw", maxHeight: "60vh" }}>
                <Grid row={true} expanded={true} justify="center">
                    {this.state.population.map(
                        (obj, index) => <IndividualButton style={{ width: individualWidth, maxHeight: "30vh", maxWidth: "30vh" }} key={index} individual={obj}></IndividualButton>)}
                </Grid>
            </div>
            <div className={styles.controlPanel}>
                {this.state.loading ? <LoadingSpinner /> : <>
                    <Grid style={{ maxWidth: gridWidth / 4 }} justify="center">
                        <PreviousGenerationButton loading={this.state.loading} onClick={this.previousGenerationClicked} />
                        <NextGenerationButton loading={this.state.loading} onClick={this.nextGenerationClicked} />
                        <SaveImagesButton loading={this.state.loading} onClick={this.saveImagesClicked} />
                    </Grid>
                </>}
            </div>
            <ToastContainer />
        </>
        );
    }
}
export default PopulationGrid;
