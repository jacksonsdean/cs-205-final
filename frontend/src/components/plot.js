import React from 'react';

import Plot from 'react-plotly.js';


class Plot extends React.Component {

  render() {

    return (

      <Plot
        id="plot"

        data={[

          {

            x: [1, 2, 3],

            y: [2, 6, 3],

            type: 'scatter',

            mode: 'lines+markers',

            marker: {color: 'red'},

          },


        ]}

        layout={ {width: 320, height: 240, title: 'Loss over time'} }

      />

    );

  }

}