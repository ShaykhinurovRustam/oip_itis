import aiohttp
import asyncio
import aiofiles
import shutil

from http import HTTPStatus


class Crawler:
    
    def __init__(self, urls: list[str]):
        self.urls = urls

    async def get_page(
        self,
        url: str, 
        session: aiohttp.ClientSession, 
        num: int,
    ) -> tuple[str, str] | None:
        async with session.get(url) as response:
            if response.status != HTTPStatus.OK:
                return None
        
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None
            
            page_info = await response.text()
            file_name = f'page-{num}.html'
            
            async with aiofiles.open(f'data/{file_name}', 'w', encoding='utf-8') as file:
                await file.write(page_info)
            
            return file_name, str(response.url)
        
    async def crawl(self) -> None:
        index = []
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.get_page(url, session, num) 
                for num, url in enumerate(self.urls, start=1)
            ]
            
            results = await asyncio.gather(*tasks)
            for result in results:
                if result is not None:
                    file_name, url = result
                    index.append(f'{file_name} {url}')
                    
        async with aiofiles.open('data/index.txt', 'w', encoding='utf-8') as file:
            await file.write('\n'.join(index))
            
        shutil.make_archive('data', 'zip', 'data')

       
async def main():
    crawler = Crawler(
        urls=[
            f'https://ru.wikipedia.org/wiki/Special:Random/{page}'
            for page in range(1, 101)
        ]
    )
    await crawler.crawl()


if __name__ == '__main__':
    asyncio.run(main())