---
layout: page
title: "Blog"
description: "Miscellaneous food for thoughts"
header-img: "img/blog-bg.jpg"
order: 3
comments: false
---

Unfortunately still under construction :sleepy:. I'm currently focusing on the [Machine Learning Glossary](/machine-learning-glossary/){:.mdLink}.

Check back soon though, I have lots of blog post ideas ! 

{% for post in site.categories.blog %}
<div class="post-preview">
    <a href="{{ post.url | prepend: site.baseurl }}">
        <h2 class="post-title">            {{ post.title }}
        </h2>
        {% if post.subtitle %}
        <h3 class="post-subtitle">
            {{ post.subtitle }}
        </h3>
        {% endif %}
    </a>
    <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
    <p class="description">{% if post.description %}{{ post.description | strip_html | strip_newlines | truncate: 250 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 250 }}{% endif %}</p>
</div>
<hr>
{% endfor %}


  