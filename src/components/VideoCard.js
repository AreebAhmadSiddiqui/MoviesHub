import React from 'react'

function VideoCard({ item, video_key, name, site }) {
    return (
        <div className='video-card'>
            <div className='video-container'>
                <iframe width="400" height="225" src={`https://www.youtube.com/embed/${video_key}`} title={name} frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                
            </div>
        </div>
    )
}

export default VideoCard