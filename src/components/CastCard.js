import React from 'react'

function CastCard({ name, profile, character }) {
    return (
        <div className='cast-card'>
            <img src={'https://image.tmdb.org/t/p/original' + profile} />
            <div><p>{name}</p>
                <p>{character}</p></div>

        </div>
    )
}

export default CastCard