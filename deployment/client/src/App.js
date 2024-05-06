import './App.css';

function App() {

  function handleUpload(e) {

    e.preventDefault();

    const formData = new FormData();
    formData.append('files[]', e.target.files[0]);


    fetch('/api/upload', {
        method: 'POST',
        body: formData
      }).then(res => {
        res.json().then(res => {console.log(res)})
      })
  }

  return (
    <div className="App">
      <form>
          <input onChange={handleUpload} type="file" name="image" />
        </form>
    </div>
  )
}

export default App;
