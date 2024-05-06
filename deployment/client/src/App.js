import './App.css';
import React, { useState } from 'react';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [file, setFile] = useState();

  function HandleUpload(e) {
    
    e.preventDefault();
    setFile(URL.createObjectURL(e.target.files[0]));
    
    const formData = new FormData();
    formData.append('files[]', e.target.files[0]);

    fetch('/api/upload', {
        method: 'POST',
        body: formData
      }).then(res => {
        res.json().then(res => {
          console.log(res.predictions);
          setPredictions(res.predictions)
        })
      })
  }

  return (
    <div className="App">
      <form>
          <input onChange={HandleUpload} type="file" name="image" />
      </form>
      <img src={file}/>
      <ul>
          {predictions.map((prediction, index) => {
            return <li key={index}>{prediction}</li>
          })}
      </ul>
    </div>
  )
}

export default App;
