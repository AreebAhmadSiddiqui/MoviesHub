import React from 'react'
import { Link } from 'react-router-dom'

function MovieCard({title,poster_path,overview,release_date,vote_average,id,media_type}) {

  return (
    <Link to={`/${media_type}/${id}`} className='movie-card' onClick={() => {
      window.scrollTo({top: 0, left: 0, behavior: 'smooth'});
    }}>
        <img className='movie-card-image' src={poster_path}/>
        <div className='movie-data'>
            <h3>{title}</h3>
            <p>{overview ? overview.substring(0,100)+"..." : ""}</p>
            <div className='movie-data-footer'>
                <span>{release_date ? release_date.split('-')[0] : ""}</span>
                <span style={{'fontWeight':'bold'}}>{vote_average.toFixed(1)}</span>
            </div>
        </div>
    </Link>
  )
}

export default MovieCard