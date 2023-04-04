import React from 'react'
import MovieCard from './MovieCard'
import HomepageSuggestion from './HomepageSuggestion'
import TrendingCarousel from './TrendingCarousel'
import Test from './Test'

function HomePage() {
  return (
    <div className='suggestions-box'>
        <TrendingCarousel/>
        <HomepageSuggestion
        title='Upcoming Movies'/>
        <HomepageSuggestion
        title='Popular Movies'/>
        <HomepageSuggestion
        title='Top Rated Movies'/>
        <HomepageSuggestion
        title='Popular TV Shows'/>
        <HomepageSuggestion
        title='Top Rated TV Shows'/>
        {/* <Test/> */}
    </div>
  )
}

export default HomePage