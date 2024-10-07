# Online Interactive Evolutionary Gallery
*Project by Jackson Dean for UVM CS 205*
    <br/>
    Available online at: <a href="https://interactive.evolutionary.gallery">interactive.evolutionary.gallery</a>
    <h2>Instructions to run locally</h2>
    <h3>With Docker</h3>
    `docker-compose up` in the project directory
    <br/><br/>
    or with newer versions of Docker: `docker compose up -d`
    <br/><br/>
    <br/>Then navigate to <a href="http://localhost:3000">localhost:3000</a>
    <h3>Without Docker</h3> Run the frontend from the "frontend" directory with <br/>`npm i`<br/>`npm run start-local`
    <br/>
    and the backend from the "backend" directory with <br/>`python -m pip install -r requirements.txt` <br/>`python local_server.py`
    <br/>Once the backend and frontend are running, navigate to <a href="http://localhost:3000">localhost:3000</a>
    <br/><br/>
    <h2>Instructions</h2>
    <b>Interactive Evolutionary Gallery</b> is a visual art project inspired
    by <a target="_blank" href="https://en.wikipedia.org/wiki/Evolutionary_algorithm">Evolutionary Algorithms</a> and <a target="_blank" href="http://eplex.cs.ucf.edu/papers/secretan_ecj11.pdf">Picbreeder</a>.
    <br /><br />
    The images represent an <i>individual</i> in a <i>population</i> of <i>genomes</i>. The population <i>evolves</i> over <i>generations</i> as some individuals survive to reproduce and others do not.
    <br /><br />
    Genomes are <i>Compositional Pattern Producing Networks (<a target="_blank" href="https://en.wikipedia.org/wiki/Compositional_pattern-producing_network">CPPNs</a>)</i>.
    A CPPN contains nodes and connections and is evaluated similar to an artificial neural network to produce an image called the <i>phenotype</i>. Nodes contain activation functions, which determine how that node adds to the phenotype.
    Connections connect nodes and have weights, which determine how much the node at the beginning of the connection contributes to the image.
    <br/><br/>
    Your job is to decide which individuals survive to reproduce. Click to select your favorite images in the population and then click <b>Next generation</b> to see the results. The individuals that you selected
    will survive unchanged in the next generation. The rest of the population is replaced with the offspring of the survivors.
    <br/><br/>
    Some offspring are produced via <i>asexual reproduction</i>. The remainder are produced via <i>crossover</i>.
    Asexual reproduction produces offspring through <i>mutation</i> of an individual's genome. Crossover combines the genomes of two parents to create a new genome.
    Choose the ratio of offspring created by crossover vs. asexual reproduction in the settings.
    <br/><br/>
    Mutations are small changes to a genome during reproduction. In the case of a CPPN, we can add connections or nodes to the network or remove them from the network.
    Adding nodes/connections increases the complexity of the phenotype image, while removing them reduces complexity. Mutations can also change a node's activation function or the weight of a connection. You can change the rate of the various mutations in the settings.
    <br/><br/>
    If you advance a generation and decide you'd rather go back to try again, press the <b>Previous generation</b> button. Images can be saved to your computer at a higher resolution by selecting the individuals and pressing <b>Save images</b>.
    Press <b>Reset</b> to start over with a new random population.


# CI/CD Pipeline
- Two branches: main and development
- Githooks prevent broken code from being pushed to main
    - Both warnings and errors cause hooks to fail on main branch
    - Only errors cause hooks to fail on development branch
    - Commits on development branch trigger tests
    - Commits on main branch trigger both build and tests
- Main branch is periodically rebased with development branch
- Mirroring repository from GitHub to GitLab
- GitLab runs CI/CD pipeline with build and tests, emails on failure

## Front-end
- Pre-commit hook checks for code complexity using xenon (built on radon) and fails if complexity rating is worse than B (absolute) A (modules) or A (average)
- Pre-commit hook checks for code quality using pylint and fails if complexity rating is worse 8.0
- Amazon Web Services (AWS) Amplify pulls from GitHub repo
- AWS builds and deploys repo (main branch)
- Pull requests on main trigger a preview build on AWS

## Back-end
- GitHub Action pushes changes to the python server code to AWS


![](./ci_cd_pipeline.png)
