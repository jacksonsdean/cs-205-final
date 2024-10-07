import React, { useState } from "react";
import styles from "./ClipText.module.css"

function ClipText() {
  // The component allows the user to input text for CLIP embeddings. 
  /// The text is used to guide the evolution of the images.

    // State variables
    const [text, setText] = useState("");

    // Event handlers
    const handleTextChange = (event) => {
        setText(event.target.value);
    };

    // JSX
    return (
        <div className={styles.container}>
            <p>Guide evolution with text:</p>
            <textarea
                id="clip-text"
                rows="1"
                cols="50"
                charswidth="23"
                className={styles.textarea}
                placeholder="Enter CLIP target"
                value={text}
                onChange={handleTextChange}
            />
        </div>
    );

  }
export default ClipText;
