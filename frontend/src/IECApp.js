import './IECApp.css';
import PopulationGrid from './components/populationGrid';
import Instructions from 'components/instructions';

function IECApp() {
  return (
    <div className="IECApp">
      <div className="IECApp-background">
        <header>
          <p>
            Interactive Evolutionary Art
          </p>

        </header>
        <div className='content'>
          <PopulationGrid />
        </div>
          <Instructions />

      </div>
    </div>
  );
}

export default IECApp;
