import './App.css';
import axios from 'axios'

function App() {

  function handleUpload(e) {

    e.preventDefault();
    console.log(e.target[0].files[0]);
    console.log(e.target[0].files);

    const formData = new FormData();
    formData.append('files[]', e.target[0].files[0]);

    const Upload = async() => {
      await fetch('/api/upload', {
        method: 'POST',
        body: formData
      }).then(res => {
        res.json().then(res => {console.log(res)})
      })
    }
    Upload();
  }

  return (
    <div className="App">
      <form>
          <input id="image" type="file" name="image" />
          <button onClick={handleUpload} type="submit" value="Submit">Submit</button>
        </form>
    </div>
  )
}

export default App;
