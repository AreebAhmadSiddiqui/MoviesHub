import React, { useEffect, useState } from 'react'
import homePageDateApi from '../config/homePageSuggestionApi'
import axios from 'axios'
import AliceCarousel from 'react-alice-carousel'
import MovieCard from './MovieCard'
import CarouselComponent from './CarouselComponent'
import temData from './temData'
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css";
import Slider from "react-slick";

function TrendingCarousel() {

    const [trendingData, setTrendingData] = useState([])
    const url = homePageDateApi['Trending Carousel']
    useEffect(() => {

        const fetchContent = async () => {
            const res = await axios.get(url)
            setTrendingData(res.data.results)
        }
        fetchContent()

    }, [])

    const settings = {
        infinite: true,
        speed: 4000,
        autoplaySpeed: 4000,
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true
      };
    
      const items = trendingData.map((item) => {
        
        return item.media_type === 'movie' ? (
                <CarouselComponent
                    title={item.title}
                    poster_path={'https://image.tmdb.org/t/p/original' + item.poster_path}
                    backdrop_path={'https://image.tmdb.org/t/p/original' + item.backdrop_path}
                    overview={item.overview}
                    release_date={item.release_date}
                    vote_average={item.vote_average}
                    key={item.id}
                    id={item.id}
                    media_type={item.release_date ? 'movie' : 'tv'}
                /> )  : (
                <CarouselComponent
                    title={item.name}
                    poster_path={'https://image.tmdb.org/t/p/original' + item.poster_path}
                    backdrop_path={'https://image.tmdb.org/t/p/original' + item.backdrop_path}
                    overview={item.overview}
                    release_date={item.first_air_date}
                    vote_average={item.vote_average}
                    key={item.id}
                    id={item.id}
                    media_type={item.release_date ? 'movie' : 'tv'}
                />)
    })
    return (
        <div className='carousel'>
            <Slider {...settings}>{items}</Slider>
        </div>
    )
}

export default TrendingCarousel