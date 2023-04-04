import React from 'react'
import homePageDateApi from '../config/homePageSuggestionApi'
import { useState } from 'react'
import { useEffect } from 'react'
import axios from 'axios'
import MovieCard from './MovieCard'

import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

function HomepageSuggestion({ title }) {
    
    const url = homePageDateApi[title]

    const [itemNumbers, setItemNumbers] = useState(6)
    const [contentData, setContentData] = useState([])

    useEffect(() => {

        const fetchContent = async () => {
            const res = await axios.get(url)
            setContentData(res.data.results)
        }
        fetchContent()

    }, [])

    function handleClick(){
        if(itemNumbers===18) setItemNumbers(6)
        else setItemNumbers(itemNumbers+6)
    }
    const data = contentData.map((item) => {
        
        return (
            <MovieCard
                title={item.title ? item.title : item.name}
                poster_path={'https://image.tmdb.org/t/p/original' + item.poster_path}
                backdrop_path={'https://image.tmdb.org/t/p/original' + item.backdrop_path}
                overview={item.overview}
                release_date={item.release_date ? item.release_date : item.first_air_date}
                vote_average={item.vote_average}
                key={item.id}
                media_type={item.release_date ? 'movie' : 'tv'}
                id={item.id}
            />
        )
    })
    return (
        <div className='outer-container'>
            <h3 style={{ 'marginBottom': '10px' }}>{title}</h3>
            <div className='inner-container'>
                {data.slice(0,itemNumbers)}
            </div>
            <div className='show-more-btn-container'>
                <button onClick={handleClick} className='show-more-btn'>{itemNumbers!==18 ? 'Show More' : 'Hide'}</button>
            </div>
        </div>
    )
}

export default HomepageSuggestion