function changeEffect(effect) {
    const videoFeed = document.getElementById('video-feed');
    if (effect === 'none') {
        videoFeed.src = "{{ url_for('video_feed') }}";
    } else {
        videoFeed.src = "{{ url_for('video_feed_effect', effect=effect) }}";
    }
}