import React from 'react';

import Plot from 'react-plotly.js';


class LossPlot extends React.Component {
    constructor(props) {
        super(props);
        // this.state = {
            // record: [],
        // }
    }
    render() {
       return <div>
        {this.props.record ? <Plot
            data={[
                {
                    x: Array.from(Array(this.props.record ? this.props.record.length : 0).keys()),
                    y: this.props.record || [],
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: {color: 'red'},
                },
            ]}
            
            layout={ {  width: 500,
                        height: 300,
                        title: 'CLIP Loss',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {color: 'white'},
                        yaxis: {
                            range: this.props.record.length==1?[0, 1]:'auto',
                            
                        }
                    } }
                    style={{maxWidth: "50vw"}}
                    /> : <></>
                }
                </div>
    }

}

export default LossPlot;