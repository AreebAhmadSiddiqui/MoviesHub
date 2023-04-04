import React from 'react'
import { Link } from 'react-router-dom'
import temData from './temData'

function CarouselComponent({title,backdrop_path,poster_path,overview,release_date,vote_average,id,media_type}) {
    return (
        <Link className='carousel-component' to={`/${media_type}/${id}`}>
            <img className='carousel-banner' src={backdrop_path} alt={title}/>
            <div className='carousel-data'>
                <div className='carousel-description'>
                    <h1>{title}</h1>
                    <p>{overview}</p>
                </div>
                <img className='carousel-img' src={poster_path} />
            </div>
        </Link>
    )
}

export default CarouselComponent