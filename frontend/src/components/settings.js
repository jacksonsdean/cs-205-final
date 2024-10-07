import React from "react";
import ReactSlider from 'react-slider'
import styles from "./Settings.module.css"
import styled from "styled-components";
import 'react-pro-sidebar/dist/css/styles.css';
import { ProSidebar, Menu, MenuItem, SubMenu } from 'react-pro-sidebar';

import { DEFAULT_CONFIG } from 'Constants';

// Custom styling for the settings menu:
const StyledProSidebar = styled(ProSidebar)`
position: fixed;
top: 0;
right:0;
height:fit-content;
width: 300px;
white-space: pre;
transition: width 4s;
max-height: 100vh;
`;

const StyledMenu = styled(Menu)`
background-color: rgba(200,200,255,.1);
border-radius: 1%;
max-height: 100vh;

`;

const StyledDropDown = styled.select`
margin-left:10px;
`;
const ResetButton = styled(MenuItem)`
color:rgba(255,155,155,.8);
margin-left:10px;
text-align:right;
`;

const Note = styled.p`
    color:rgba(255,255,255,.3);
    text-align:right;
    margin:none;
    padding:none;
    margin-right:10px;
    text-decoration:none;
    font-style: italic;
`;
// end custom styling


function newSetting(settings, setting, value, popGrid, requiresRestart = false) {
    /*A new setting was chosen, save it and update the population grid if necessary*/
    let newSetting = { ...settings };
    newSetting[setting] = value;
    popGrid.current.setSettings(newSetting)
    if (requiresRestart) {
        popGrid.current.reset()
    }
    return newSetting;
}

function ProbabilitySlider(props) {
    /*A slider for choosing a probability value*/
    return <ReactSlider
        min={0}
        max={1}
        step={.01}
        value={props.value}
        onAfterChange={props.onAfterChange}
        className={styles.horizontalSlider}
        thumbClassName={styles.thumb}
        trackClassName={styles.track}
        renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
    />
}


function IntField(props) {
    return <td><input type="number" step="1" value={props.value} onChange={(event) => props.onAfterChange(parseInt(event.target.value))} /></td>
}
function FloatField(props) {
    return (<td>
            <input
                type='number'
                step="0.1"
                min='0'
                className='form-control'
                value={props.value}
                onChange={(event) => props.onAfterChange(parseFloat(event.target.value))}
            />
        </td>)
}

function Settings(props) {
    /*The settings component that allows the user to change settings*/
    const settings = props.loadedSettings;
    const popGrid = props.popGrid;
    return (
        <form>
            <StyledProSidebar>
                <StyledMenu iconShape="round">
                    <SubMenu title="Settings">
                        <SubMenu title="General">
                            <MenuItem>
                                Radial symmetry bias *<input type="checkbox" checked={settings.use_radial_distance} onChange={(event) => props.setSettingsCallback(newSetting(settings, "use_radial_distance", event.target.checked, popGrid, true))} />
                            </MenuItem>
                            <MenuItem>
                                {"Color mode *"}
                                <StyledDropDown value={settings.color_mode} onChange={(event) => props.setSettingsCallback(newSetting(settings, "color_mode", event.target.value, popGrid, true))}>
                                    <option value="L">Grayscale</option>
                                    <option value="HSL">HSL</option>
                                    <option value="RGB">RGB</option>
                                </StyledDropDown>
                            </MenuItem>
                            <MenuItem>Offspring crossover ratio
                                <ProbabilitySlider
                                    value={settings.prob_crossover}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_crossover", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem># CLIP SGD Steps
                                <IntField
                                    value={settings.sgd_steps}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "sgd_steps", value, popGrid)) }}
                                />
                            </MenuItem>

                            <MenuItem># CLIP SGD LR
                                <FloatField
                                    value={settings.sgd_learning_rate}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "sgd_learning_rate", value, popGrid)) }}
                                />
                            </MenuItem>

                            <Note>{"*requires restart"}</Note>

                        </SubMenu>

                        <SubMenu title="Mutation rates">
                            <MenuItem>Weights
                                <ProbabilitySlider
                                    value={settings.prob_mutate_weight}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_mutate_weight", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem>Add nodes
                                <ProbabilitySlider
                                    value={settings.prob_add_node}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_add_node", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem>Remove nodes
                                <ProbabilitySlider
                                    value={settings.prob_remove_node}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_remove_node", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem>Add connections
                                <ProbabilitySlider
                                    value={settings.prob_add_connection}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_add_connection", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem>Remove connections
                                <ProbabilitySlider
                                    value={settings.prob_disable_connection}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_disable_connection", value, popGrid)) }}
                                />
                            </MenuItem>
                            <MenuItem>Change activation function
                                <ProbabilitySlider
                                    value={settings.prob_mutate_activation}
                                    onAfterChange={(value) => { props.setSettingsCallback(newSetting(settings, "prob_mutate_activation", value, popGrid)) }}
                                />
                            </MenuItem>

                        </SubMenu>
                        <ResetButton onClick={() => {
                            popGrid.current.setSettings(DEFAULT_CONFIG)
                            popGrid.current.reset()
                            props.setSettingsCallback(DEFAULT_CONFIG);
                        }}>Reset all to default</ResetButton>
                    </SubMenu>
                </StyledMenu>
            </StyledProSidebar>
        </form>
    );
}
export default Settings;
