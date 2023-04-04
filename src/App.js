import './App.css';
import Navbar from './components/Navbar';
import HomePage from './components/HomePage';
import Footer from './components/Footer';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import MoviesPage from './components/MoviesPage';
import TvPage from './components/TvPage'
import DetailsPage from './components/DetailsPage'
function App() {
  return (
    <BrowserRouter>
      <div className='app'>
        <Navbar />
        <Routes>
          <Route path='/' element={<HomePage/>} exact/>
          <Route path='/movie/:id' element={<DetailsPage/>} />
          <Route path='/tv/:id' element={<DetailsPage/>} />
          <Route path='/movies' element={<MoviesPage/>} exact/>
          <Route path='/tv' element={<TvPage/>} exact/>
        </Routes>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
