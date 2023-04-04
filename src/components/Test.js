import React from 'react'
import axios from 'axios'
import { useState, useEffect } from 'react'
import { type } from '@testing-library/user-event/dist/type'
import MovieCard from './MovieCard'
function Test() {

    const [predData, setPredData] = useState([])
    const [content, setContent] = useState([])

    const id=640146
    const url = 'http://localhost:5000/predict'

    const fetchMovie = async (id) => {

        console.log(id);
        const movieUrl=`https://api.themoviedb.org/3/movie/${id}?api_key=8f655507141a5d524fc2024c9f76b6c7&language=en-US`
        const res = await axios.get(movieUrl)

        // console.log(res.data);
        content.push(res.data)
        setContent(content)
    }

    useEffect(() => {
        const fetchPredData = async () => {
            const res = await axios.post(url, { "id": 274854 })
            var a = res.data.prediction;
            a = a.replace(/'/g, '"');
            a = JSON.parse(a);
            setPredData(a)
        }
        fetchPredData()

        
    }, [])

    useEffect(()=>{
        
        predData.forEach((movie_id)=>{
            fetchMovie(movie_id)
        })

    },[predData])


    const data = content.map((item) => {
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
        <div className='movies-container'> {data} </div>
    )
}

export default Test