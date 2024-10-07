import './IECApp.css';
import React, { useState } from 'react';
import PopulationGrid from './components/populationGrid';
import Instructions from 'components/instructions';
import Settings from 'components/settings';
import { DEFAULT_CONFIG } from 'Constants';
import ClipText from 'components/clipText';


/*Functions to save and load settings from browser local storage */
function saveSettings(settings) {
  // Save settings to local storage
  for (let key in settings) {
    localStorage.setItem(key, settings[key])
  }
}
function loadSettings() {
  // Load settings from local storage
  let settings = {...DEFAULT_CONFIG}; // start with default settings
  for (let key in settings) {
    let value = localStorage.getItem(key);
    if (value !== null) {
      // convert to array if possible
      if (value.includes(',')) {
        value = value.split(',');
      }
      // convert to number if possible
      if (value.length>0 && Number.isFinite(Number(value))) {
        value = Number(value);
      }
      // convert to boolean if possible
      if (value === 'true') {
        value = true;
      }
      if (value === 'false') {
        value = false;
      }
      settings[key] = value;
    }
  }
  return settings
}

function IECApp() {
  // The main component of the app, loads all other components and manages settings state
  let [settings, setSettingsState] = useState(loadSettings());
  let popGrid = React.createRef();
  const setSettings = (newSettings) => {
    setSettingsState(newSettings);
    saveSettings(newSettings);
  }

  return (
    <div className="IECApp">
      <div className="IECApp-background">
        <header>
          <p>
            Interactive Evolutionary Gallery
          </p>

        </header>
        <div className='content'>
          <PopulationGrid ref={popGrid} settings={settings}/>
          <ClipText />
        </div>
          <Instructions />
        <Settings loadedSettings={settings} setSettingsCallback={setSettings} popGrid={popGrid}/>
      </div>
    </div>
  );
}

export default IECApp;
