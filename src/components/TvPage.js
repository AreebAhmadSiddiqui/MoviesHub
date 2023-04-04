import React from 'react'
import { useEffect, useState } from 'react'
import { BsSearch } from "react-icons/bs";
import axios from 'axios';
import MovieCard from './MovieCard';
function ContentPage() {


    const [click,setClick]=useState(0);
    const [pageNo, setPageNo] = useState(1);
    const [contentData, setContentData] = useState([])
    const [searchContent, setSearchContent] = useState("")
    const [filteredData,setFilteredData]=useState([])


    const url = `https://api.themoviedb.org/3/tv/on_the_air?api_key=8f655507141a5d524fc2024c9f76b6c7&language=en-US&page=${pageNo}`

    useEffect(() => {

        const fetchContent = async () => {
            const res = await axios.get(url)
            setContentData(contentData.concat(res.data.results))
        }
        fetchContent()

    }, [pageNo])


    function handleClick() {
        setPageNo(prevPageNo => prevPageNo + 1)
    }
    const data = click==1 ? filteredData.map((item) => {

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
    }) : contentData.map((item) => {

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

    
    
    const handleClick2 = async () => {
        
        setClick(1)
        const url= `https://api.themoviedb.org/3/search/tv?api_key=8f655507141a5d524fc2024c9f76b6c7&language=en-US&page=1&query=${searchContent}&include_adult=false`
        const res = await axios.get(url)
        setFilteredData(res.data.results);
    };
    
    function handleChange(event) {
        setSearchContent(event.target.value)
    }

    return (
        <div className='movie-page-container'>
            <div className='input-container'>
                <input
                    type='text'
                    placeholder='Search movie'
                    onChange={handleChange}
                    name='searchContent'
                    value={searchContent} />
                <button className='btn' onClick={handleClick2}><BsSearch className='search-btn-icon' /></button>
            </div>

            <div className='movie-list-container'>
                <div className='movies-container'>
                    {data}
                </div>
                <div className='show-more-btn-container'>
                    {click==0 ? <button className='show-more-btn' onClick={handleClick}>Show More</button> : <></>}
                </div>
            </div>
        </div>
    )
}

export default ContentPage