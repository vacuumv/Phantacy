import datetime
import io
import json
import re

import requests


class DatetimeConverter:
    time_format = "%Y-%m-%dT%H:%M:%S+0000"
    output_time_format = "%Y-%m-%d %H:%M:%S"
    date_format = "%Y-%m-%d"

    @staticmethod
    def from_time_string(time_string, to_string=True):
        time = datetime.datetime.strptime(time_string, DatetimeConverter.time_format)
        return time.strftime(DatetimeConverter.output_time_format) if to_string else time


class Post:
    prev_created_time = None

    def __init__(self, pid, pwid, time, pcontent, fpr, pcids, plink):
        create_date = DatetimeConverter.from_time_string(time, to_string=False)
        self.post_id = pid
        self.post_writer_id = pwid
        self.post_created_time = time
        self.post_content = pcontent
        self.fan_page_reply = fpr
        self.post_comment_ids = pcids
        self.post_link = plink
        time_elapsed = Post.prev_created_time - create_date if Post.prev_created_time is not None else None
        self.time_interval = int(time_elapsed.total_seconds()) if time_elapsed is not None else None

    @staticmethod
    def set_prev_created_time(prev):
        Post.prev_created_time = DatetimeConverter.from_time_string(prev, to_string=False)

    def __repr__(self):
        fmt = u"文章 id: {post_id}  [{created_time} ({interval})]   " \
              u"[全部的評論: {comment_count}] " \
              u"[粉絲團回覆: {reply}]\n{content}\n" \
              u"連結: {link}"
        return fmt.format(post_id=self.post_id,
                          created_time=self.post_created_time,
                          interval=self.time_interval,
                          comment_count=len(self.post_comment_ids),
                          reply=self.fan_page_reply,
                          content=self.post_content,
                          link=self.post_link)


class RequestUrlGenerator:
    """
    產生Facebook Graph api Request之產生器
    """
    request_format = 'https://graph.facebook.com/{api_version}/{request_object}/' \
                     '?fields={request_fields}&access_token={facebook_access_token}'
    visitor_request_fields_format = "visitor_posts.limit(1).order(chronological){request_time_interval}" \
                                    "{comments.limit(2){from},created_time,message,from,permalink_url}"
    request_time_interval_format = ".since({}).until({})"
    # 不建議設超過100 和抓取速度有關
    api_request_post_max_count = 50
    api_request_review_max_count = 50

    def __init__(self, api_version, request_object, facebook_access_token):
        self.api_version = api_version
        self.facebook_access_token = facebook_access_token
        self.request_object = request_object

    def _get_request_url(self, request_fields):
        return RequestUrlGenerator.request_format.format(api_version=self.api_version,
                                                         request_object=self.request_object,
                                                         request_fields=request_fields,
                                                         facebook_access_token=self.facebook_access_token)

    def get_visitor_request_url(self, posts_limit=-1, comments_limit=-1, since=None, until=None):
        if posts_limit > self.api_request_post_max_count or posts_limit == -1:
            posts_limit = self.api_request_post_max_count
        if comments_limit > self.api_request_review_max_count or comments_limit == -1:
            comments_limit = self.api_request_post_max_count
        request_time_str = "" if since is None or until is None \
            else self.request_time_interval_format.format(since, until)
        # print(request_time_str)
        visitor_request_fields = self.visitor_request_fields_format \
            .replace("(1)", "({})".format(posts_limit)) \
            .replace("(2)", "({})".format(comments_limit)) \
            .replace("{request_time_interval}", request_time_str)
        return self._get_request_url(visitor_request_fields)


class FanPageCrawler:
    def __init__(self, facebook_access_token, fan_page_id):
        self.facebook_access_token = facebook_access_token
        self.fan_page_id = fan_page_id
        self.post_output_list = []

    def _get_request_url(self, posts_limit=-1, comments_limit=-1, since=None, until=None):
        generator = RequestUrlGenerator(api_version='v2.10',
                                        request_object=self.fan_page_id,
                                        facebook_access_token=self.facebook_access_token)
        return generator.get_visitor_request_url(posts_limit, comments_limit, since, until)

    def crawl_posts_and_save(self, output_file_path, posts_limit=-1, comments_limit=-1, since=None, until=None):
        posts_list = self.crawl_posts(posts_limit, comments_limit, since, until)
        print('抓取完畢，寫入檔案中')
        self._write_json_to_result_file(posts_list, output_file_path)
        print('寫入檔案完畢，可開始分析')

    def crawl_posts(self, posts_limit=-1, comments_limit=-1, since=None, until=None):
        print("開始抓取資料")

        request_url = self._get_request_url(posts_limit, comments_limit, since, until)
        print(request_url)
        response = requests.get(request_url).json()

        if 'error' in response:
            print('存取以下url錯誤: \n{}'.format(request_url))
            print('錯誤訊息: {}'.format(response['error']['message']))
            exit(1)

        pid = 0
        for post in self._get_all_elements(response['visitor_posts'], '貼文查詢請求已傳送.', posts_limit):
            if 'message' in post:
                pid += 1
                post_id = str(post['id'])
                post_writer_id = int(post['from'][u'id'])
                post_created_time = str(post['created_time'])
                post_content = self._message_pre_process(post['message'])
                post_link = post.get('permalink_url', '')
                post_comments = post.get('comments', [])
                post_comment_ids = [int(comment['from'][u'id']) for comment in
                                    self._get_all_elements(post_comments,
                                                           '則評論已從貼文[{pid}]抓取'.format(pid=pid + 1),
                                                           comments_limit)] if len(post_comments) > 0 else []
                fan_page_reply = True if int(self.fan_page_id) in post_comment_ids else False
                post_comment_ids = [iid for iid in post_comment_ids if iid != int(self.fan_page_id)]

                self.post_output_list.append(
                    Post(post_id, post_writer_id, post_created_time, post_content, fan_page_reply,
                         post_comment_ids, post_link))
                Post.set_prev_created_time(post_created_time)
        for post in self.post_output_list:
            print(post)
        return self.post_output_list

    @staticmethod
    def _write_json_to_result_file(py_object, file_name):
        with io.open(file_name, 'w', encoding='utf-8') as result_file:
            data = json.dumps(py_object, default=lambda o: o.__dict__,
                              sort_keys=True, indent=4, ensure_ascii=False)
            result_file.write(data)

    @staticmethod
    def _message_pre_process(message):
        """
        將貼文去掉網址
        :param message: 貼文內容
        :return: 去掉網址的貼文內容
        """
        message = str(message).replace('\n', '')
        message = re.sub("http[s]?://[^/\s]+/[^\s]+", "", message)
        return message

    @staticmethod
    def _get_all_elements(response, message, limit):
        result_list = []
        while 'paging' in response:
            result_list.extend(response['data'])
            if len(result_list) > limit != -1:
                break
            if 'next' in response['paging']:
                next_request_url = response['paging']['next']
                # print(next_request_url)
                response = requests.get(next_request_url).json()
                print('目前 {} {}'.format(len(result_list), message))
            else:
                break
        number = len(result_list) if limit == -1 else limit
        print("總共 {} {}".format(number, message))
        return result_list[:number]


def main():
    # 貼文抓取數量
    post_limit = -1
    # 評論抓取數量
    comment_limit = -1
    #
    since = '2017-12-20'
    until = '2017-12-25'
    # 輸出檔案位置
    output_file_path = "/Users/Steve/PycharmProjects/Phantacy/result_all.json"
    # 粉絲團ID
    fan_page_id = "818773044827025"
    # 存取權杖
    access_token = "EAACEdEose0cBABt6nx1UQZA4ciatYURIWDcVQ5sJZA1GKCoJCK4J2f5RAlDWy5TZChDvOrJS9gVy1j6r6vcrBYGCGX6VzLhe7LVtaSdbUh4RmmO8anYCr9XlSyayRZAG2ah71LwnkxygGenGMtBhHyLARZBdN40w14BgxwSSD5ClM2AnEKUedljvaGTNGy1ApBOp9AemJ5wZDZD"

    crawler = FanPageCrawler(facebook_access_token=access_token,
                             fan_page_id=fan_page_id)

    posts = crawler.crawl_posts(
        # output_file_path=output_file_path,
        posts_limit=post_limit,
        comments_limit=comment_limit,
        since=since,
        until=until)
    for post in posts:
        print(post)


if __name__ == '__main__':
    main()
